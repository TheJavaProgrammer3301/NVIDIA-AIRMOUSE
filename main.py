import os
import sys
import pyautogui
import numpy as np
import time
import queue
import threading
import keyboard
import argparse
from collections import deque
from typing import Optional, Tuple
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, cudaAllocMapped

class HandKeypoints:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

# parse the command line
parser = argparse.ArgumentParser(description="Control your mouse with your hand", 
                                 formatter_class=argparse.RawTextHelpFormatter)

# based on jetson inference posenet example
parser.add_argument("--frequency", type=float, default=30, help="update frequency for the mouse (default: 30 fps)")
parser.add_argument("--multiplier", type=float, default=2, help="multiplier for finger position relative to center of camera (default: 2)")
parser.add_argument("--keypoint", type=int, default=2, help="keypoint to track (2 for wrist, 8 for index finger tip, etc.)\nvalid values are: 0-20 (see HandKeypoints class for details)")
parser.add_argument("--click-threshold", type=int, default=3, help="number of sequential frames where a click gesture is detected for a change in clicking to occur (default: 3)")
parser.add_argument("--overlay", type=bool, default=False, help="whether to enable overlay (default: false)")

try:
	args = parser.parse_known_args()[0]
except:
	print("bad args")
	parser.print_help()
	sys.exit(0)

pose_net = None

FRAMERATE = args.frequency

CLICK_CHANGE_THRESHOLD = args.click_threshold
SINGLE_CLICK_THRESHOLD = CLICK_CHANGE_THRESHOLD * 2

POSITION_MULTIPLIER = args.multiplier # multiplier for finger position relative to center of camera
TRACKED_KEYPOINT = args.keypoint

# wip flags
ENABLE_PHYSICAL_MOUSE_SIM = False

MAX_MOUSE_ACCEL = 4 # max mouse acceleration, fraction of screen size
MAX_MOUSE_VEL = 4 # max mouse velocity, fraction of screen size

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
pyautogui.DARWIN_CATCH_UP_TIME = 0

def vec_mag(x: float, y: float) -> float:
    return np.sqrt(x**2 + y**2)

def limit_mag(x: float, y: float, max_mag: float) -> Tuple[float, float]:
    mag = vec_mag(x, y)
    if mag > max_mag:
        x /= mag / max_mag
        y /= mag / max_mag
    return x, y

class ProcessingResult:
    poses: any
    finger_tip: Optional[any] = None
    processing_time: float = 0.0
    frame_id: int = 0
    click_type: int = 0
    
    def __init__(self, poses, frame_id):
        self.poses = poses
        self.finger_tip = None
        self.processing_time = None
        self.frame_id = frame_id
        self.click_type = 0
    
class KeypointNames:
    MAP = {
        0: "palm",
        1: "thumb 1",
        2: "thumb 2",
        3: "thumb 3",
        4: "thumb tip",
        5: "index finger 1",
        6: "index finger 2",
        7: "index finger 3",
        8: "index finger tip",
        9: "middle finger 1",
        10: "middle finger 2",
        11: "middle finger 3",
        12: "middle finger tip",
        13: "ring finger 1",
        14: "ring finger 2",
        15: "ring finger 3",
        16: "ring finger tip",
        17: "baby finger 1",
        18: "baby finger 2",
        19: "baby finger 3",
        20: "baby finger tip"
    }

class GestureClickDetector:
    """Detect fist gestures for clicking"""
    
    def __init__(self):
        # Gesture thresholds
        self.fist_threshold = 80   # pixels
        
        # Gesture history for stability
        self.gesture_history = deque(maxlen=5)
        self.last_gesture_state = False
        
        self.frames_with_gesture = 0
        self.frames_without_gesture = 0
        self.click_length = 0
        
        # Calibration
        self.hand_scale = 1.0
        self.calibration_samples = deque(maxlen=30)
    
    def calculate_distance(self, point1, point2) -> float:
        """Calculate Euclidean distance between two keypoints"""
        if point1 is None or point2 is None:
            return float('inf')
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def auto_calibrate_hand_size(self, keypoints):
        """Auto-calibrate thresholds based on hand size"""
        wrist = None
        middle_tip = None
        
        for kp in keypoints:
            if kp.ID == HandKeypoints.WRIST:
                wrist = kp
            elif kp.ID == HandKeypoints.MIDDLE_TIP:
                middle_tip = kp
        
        if wrist and middle_tip:
            hand_length = self.calculate_distance(wrist, middle_tip)
            self.calibration_samples.append(hand_length)
            
            if len(self.calibration_samples) >= 20:
                avg_hand_length = np.mean(self.calibration_samples)
                # Scale thresholds based on hand size (normalized to ~150 pixel hand)
                self.hand_scale = avg_hand_length / 150.0
                self.fist_threshold = 80 * self.hand_scale
    
    def detect_fist(self, keypoints) -> bool:
        """Detect closed fist (all fingertips close to palm center)"""
        wrist = None
        fingertips = []
        palm_center = None
        
        # Get key points
        for kp in keypoints:
            if kp.ID == HandKeypoints.WRIST:
                wrist = kp
            elif kp.ID in [HandKeypoints.INDEX_TIP, HandKeypoints.MIDDLE_TIP, HandKeypoints.RING_TIP, HandKeypoints.PINKY_TIP]:
                fingertips.append(kp)
            elif kp.ID == HandKeypoints.MIDDLE_MCP:  # Use middle MCP as palm center reference
                palm_center = kp
        
        if not wrist or len(fingertips) < 3:
            return False
        
        # Use palm center if available, otherwise use wrist
        reference_point = palm_center if palm_center else wrist
        
        # Calculate average distance of fingertips to palm
        distances = [self.calculate_distance(tip, reference_point) for tip in fingertips]
        avg_distance = np.mean(distances)
        
        # print(f"fist: {avg_distance / self.fist_threshold}")
        # Check if fingers are curled (close to palm)
        return avg_distance < self.fist_threshold
    
    def detect_click(self, pose) -> int:
        """Main click detection method"""

        if not pose or not pose.Keypoints:
            return 0

        self.auto_calibrate_hand_size(pose.Keypoints)

        gesture_detected = self.detect_fist(pose.Keypoints)

        # Update gesture state counters
        if gesture_detected:   
            self.frames_with_gesture += 1
            self.frames_without_gesture = 0
            self.click_length += 1
        else:
            self.frames_without_gesture += 1
            self.frames_with_gesture = 0

        # Transition logic
        if self.frames_with_gesture >= CLICK_CHANGE_THRESHOLD:
            self.last_gesture_state = True
            
            return 1
        elif self.last_gesture_state and self.frames_without_gesture >= CLICK_CHANGE_THRESHOLD:
            self.last_gesture_state = False
            
            if self.click_length <= SINGLE_CLICK_THRESHOLD:
                return 2
                
            self.click_length = 0

        return 0

class MouseController:
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, 
                 camera_width: int = 640, camera_height: int = 480):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        self.center_x = camera_width // 2
        self.center_y = camera_height // 2
        
        # # Mouse movement parameters
        # self.smooth_factor = 0.4
        # self.prev_x = screen_width // 2
        # self.prev_y = screen_height // 2
        # self.movement_threshold = 5
        
        self.pos_x = self.center_x
        self.pos_y = self.center_y
        # self.vel_x = 0
        # self.vel_y = 0
  
        self.MAX_ACCEL = np.sqrt(screen_width**2 + screen_height**2) * MAX_MOUSE_ACCEL
        self.MAX_VEL = np.sqrt(screen_width**2 + screen_height**2) * MAX_MOUSE_VEL
        
        self.finger_x = self.center_x
        self.finger_y = self.center_y
        
        # Center the mouse on initialization
        self.center_mouse()
        
        self.clicking = False
    
    def center_mouse(self):
        """Center the mouse cursor on screen"""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        try:
            pyautogui.moveTo(center_x, center_y)
            self.prev_x = center_x
            self.prev_y = center_y
            print(f"Mouse centered at ({center_x}, {center_y})")
        except Exception as e:
            print(f"Failed to center mouse: {e}")
    
    def update_mouse_physical(self, finger_x: float, finger_y: float, delta_time: float) -> bool:
        screen_x = max(0, min(int((1.0 - finger_x / self.camera_width) * self.screen_width), self.screen_width - 1))
        screen_y = max(0, min(int((finger_y / self.camera_height) * self.screen_height), self.screen_height - 1))
        
        # a_x = 0
        # a_y = 0
        
        # if delta_time > 0:
        #     a_x = (screen_x - self.pos_x) / delta_time
        #     a_y = (screen_y - self.pos_y) / delta_time
        
        # a_x, a_y = limit_mag(a_x, a_y, self.MAX_ACCEL)
   
        # self.vel_x += a_x * delta_time
        # self.vel_y += a_y * delta_time
        
        # self.vel_x, self.vel_y = limit_mag(self.vel_x, self.vel_y, min(self.MAX_VEL, vec_mag(screen_x - self.pos_x, screen_y - self.pos_y)))
        
        # self.pos_x = max(0, min(self.pos_x + self.vel_x * delta_time, self.screen_width - 1))
        # self.pos_y = max(0, min(self.pos_y + self.vel_y * delta_time, self.screen_height - 1))
        
        # print(f"pos_x: {self.pos_x}, pos_y: {self.pos_y}, vel_x: {self.vel_x}, vel_y: {self.vel_y}, a_x: {a_x}, a_y: {a_y}")
        
        vel_x = 0
        vel_y = 0
        
        if delta_time > 0:
            vel_x = (screen_x - self.pos_x) / delta_time
            vel_y = (screen_y - self.pos_y) / delta_time
            
        vel_x, vel_y = limit_mag(vel_x, vel_y, min(self.MAX_VEL, vec_mag(screen_x - self.pos_x, screen_y - self.pos_y)))
        
        self.pos_x = max(0, min(self.pos_x + vel_x * delta_time, self.screen_width - 1))
        self.pos_y = max(0, min(self.pos_y + vel_y * delta_time, self.screen_height - 1))
  
        print(f"pos_x: {self.pos_x}, pos_y: {self.pos_y}, vel_x: {vel_x}, vel_y: {vel_y}, screen_x: {screen_x}, screen_y: {screen_y}")
  
        try:
            pyautogui.moveTo(self.pos_x, self.pos_y)
            return True
        except pyautogui.FailSafeException:
            print("PyAutoGUI fail-safe triggered - move mouse away from corner")
            return False
        except Exception as e:
            print(f"Mouse movement error: {e}")
            return False
    
    def update_mouse_position(self, finger_x: float, finger_y: float, delta_time: float) -> bool:
        """Update mouse position based on finger coordinates"""
        # Convert camera coordinates to screen coordinates (with mirroring)
        screen_x = int((1.0 - finger_x / self.camera_width) * self.screen_width)
        screen_y = int((finger_y / self.camera_height) * self.screen_height)
        
        # Apply smoothing
        smooth_x = int(self.prev_x + (screen_x - self.prev_x) * self.smooth_factor)
        smooth_y = int(self.prev_y + (screen_y - self.prev_y) * self.smooth_factor)
        
        # Calculate movement distance
        distance = np.sqrt((smooth_x - self.prev_x) ** 2 + (smooth_y - self.prev_y) ** 2)
        
        # Only move if above threshold
        # if distance > self.movement_threshold:
        # Clamp to screen bounds
        smooth_x = max(0, min(smooth_x, self.screen_width - 1))
        smooth_y = max(0, min(smooth_y, self.screen_height - 1))
        
        try:
            pyautogui.moveTo(smooth_x, smooth_y)
            self.prev_x = smooth_x
            self.prev_y = smooth_y
            return True
        except pyautogui.FailSafeException:
            print("PyAutoGUI fail-safe triggered - move mouse away from corner")
            return False
        except Exception as e:
            print(f"Mouse movement error: {e}")
            return False
        
        # return True
    
    def set_clicking(self, click: int):
        """Execute mouse click"""
        try:
            clicking = click == 1
            clicking_changed = clicking != self.clicking
            
            if clicking_changed:
                if clicking:
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()
                
                self.clicking = clicking
    
                print(f"clicking changed to: {clicking}")
            if click == 2:
                print("single clicking")
                pyautogui.click()
        except Exception as e:
            print(f"Click failed: {e}")
            
    def process(self, finger_x: float, finger_y: float, click: int, delta_time: float) -> bool:
        if finger_x is not None:
            self.finger_x = max(0, min(((finger_x - self.center_x) * POSITION_MULTIPLIER) + self.center_x, self.camera_width - 1))
            self.finger_y = max(0, min(((finger_y - self.center_y) * POSITION_MULTIPLIER) + self.center_y, self.camera_height - 1))
            
        if ENABLE_PHYSICAL_MOUSE_SIM:
            self.update_mouse_physical(self.finger_x, self.finger_y, delta_time)
        else:
            self.update_mouse_position(self.finger_x, self.finger_y, delta_time)
        
        if click is not None:
            self.set_clicking(click)

class FistHandTracker:
    def __init__(self, target_fps: int = 60):
        """Initialize hand tracker with gesture-based clicking"""
        # Get screen resolution
        try:
            self.screen_width, self.screen_height = pyautogui.size()
            print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            self.screen_width, self.screen_height = 1920, 1080
            print(f"Could not get screen size ({e}), using default: {self.screen_width}x{self.screen_height}")
        
        # Initialize video input/output
        print("Initializing video source...")
        self.input = videoSource("/dev/video0")
        self.output = videoOutput("webrtc://@:8554/output")
        
        # Get camera dimensions
        self.camera_width = self.input.GetWidth()
        self.camera_height = self.input.GetHeight()
        print(f"Camera resolution: {self.camera_width}x{self.camera_height}")
        
        # Initialize mouse controller
        self.mouse_controller = MouseController(
            self.screen_width, self.screen_height,
            self.camera_width, self.camera_height
        )
        
        # Click detection
        self.gesture_detector = GestureClickDetector()
        
        # Performance parameters
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Performance tracking
        self.frame_count = 0
        self.fps_timer = time.time()
        self.processing_times = deque(maxlen=60)
        
        # Frame skipping
        self.skip_counter = 0
        self.max_consecutive_skips = 3
        
        # Hand tracking parameters
        self.last_finger_pos = None
        self.finger_lost_frames = 0
        self.max_finger_lost_frames = 10
        
        # Statistics
        self.click_count = 0
        
        # Warm up the network
        self._warmup_network()
        
        self.running = False
        self.paused = False
  
        self.output_queue = queue.Queue()
        
    def toggle_program(self):
        self.paused = not self.paused
        print(f"\n{'='*60}")
        print(f"Program {'paused' if self.paused else 'resumed'}")
        print(f"{'='*60}\n")
        
    def stop_program(self):
        self.running = False
    
    def _warmup_network(self):
        """Warm up PoseNet to avoid slow first frame"""
        print("Warming up PoseNet...")
        try:
            # Create a dummy image
            dummy_img = cudaAllocMapped(
                width=self.camera_width, 
                height=self.camera_height, 
                format="rgb8"
            )
            
            # Process dummy frame
            start = time.time()
            pose_net.Process(dummy_img)
            warmup_time = time.time() - start
            
            print(f"PoseNet warmup completed in {warmup_time*1000:.1f}ms")
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")
    
    def process_frame(self, img_input) -> ProcessingResult:
        """Process a single frame through PoseNet"""
        start_time = time.time()
        
        # Create result object
        result = ProcessingResult(
            poses=None,
            frame_id=self.frame_count
        )
        
        # Process pose detection
        result.poses = pose_net.Process(img_input)
        
        # Find index finger tip and primary pose
        primary_pose = None
        finger_found = False
        
        if result.poses:
            # Use the first detected pose
            primary_pose = result.poses[0]
            
            if primary_pose.Keypoints:
                for keypoint in primary_pose.Keypoints:
                    if keypoint.ID == TRACKED_KEYPOINT:
                        result.finger_tip = keypoint
                        self.last_finger_pos = (keypoint.x, keypoint.y)
                        self.finger_lost_frames = 0
                        finger_found = True
                        break
        
        # Handle lost finger tracking
        if not finger_found and self.last_finger_pos and self.finger_lost_frames < self.max_finger_lost_frames:
            self.finger_lost_frames += 1
            # Create a synthetic keypoint at last known position
            class SyntheticKeypoint:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    self.ID = 8
            
            result.finger_tip = SyntheticKeypoint(*self.last_finger_pos)
        
        # Check for click gesture if we have a pose
        if primary_pose:
            click_type = self.check_for_click(primary_pose)
            result.click_type = click_type
        
        result.processing_time = time.time() - start_time
        return result
    
    def check_for_click(self, pose) -> int:
        return self.gesture_detector.detect_click(pose)
    
    def should_skip_frame(self) -> bool:
        """Determine if the current frame should be skipped"""
        if not self.processing_times:
            return False
        
        avg_time = np.mean(self.processing_times)
        
        # Skip if we're running significantly behind
        if avg_time > self.frame_time * 0.8:
            if self.skip_counter < self.max_consecutive_skips:
                self.skip_counter += 1
                return True
        else:
            self.skip_counter = 0
        
        return False
    
    def print_stats(self):
        """Print performance statistics"""
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            max_time = np.max(self.processing_times) * 1000
            min_time = np.min(self.processing_times) * 1000
            
            fps = self.frame_count / 2.0  # Stats printed every 2 seconds
            
            print(f"FPS: {fps:.1f} | Process time (ms): "
                  f"avg={avg_time:.1f}, min={min_time:.1f}, max={max_time:.1f} | "
                  f"Clicks: {self.click_count}")
    
    def posenet_worker(self):
        try:
            while self.running:
                # Capture frame
                img_input = self.input.Capture(format='rgb8', timeout=1000)
                if img_input is None:
                    print("Failed to capture frame")
                    continue
                
                # Check if we should skip this frame
                if self.should_skip_frame():
                    continue
                
                # Process the frame
                result = self.process_frame(img_input)
                self.processing_times.append(result.processing_time)
                
                # Update mouse position if finger is detected
                x, y = (None, None) if result.finger_tip is None else (result.finger_tip.x, result.finger_tip.y)
                
                self.output_queue.put({
                    'x': x,
                    'y': y,
                    'click': result.click_type
                })
                
                # Render output if available
                if self.output:
                    self.output.Render(img_input)
                
                # Update frame counter
                self.frame_count += 1
                
                # Print statistics every 2 seconds
                current_time = time.time()
                if current_time - self.fps_timer > 2.0:
                    self.print_stats()
                    self.frame_count = 0
                    self.fps_timer = current_time
        
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def run(self):
        """Main processing loop"""
        self.running = True
        
        keyboard.on_press_key("enter", lambda e: self.stop_program())
        keyboard.on_press_key("space", lambda e: self.toggle_program())
        
        # Print instructions
        print(f"\n{'='*60}")
        print(f"Starting PoseNet hand tracker")
        print(f"Mouse FPS: {FRAMERATE} fps")
        print("GESTURE: Make a fist to click")
        print(f"\nMove your {KeypointNames.MAP[TRACKED_KEYPOINT]} to control the mouse")
        print("Press ENTER to exit, press SPACE to pause/unpause\n")
        print(f"{'='*60}")
  
        time.sleep(2)
        
        self.mouse_controller.process(None, None, None, 0)
        
        posenet_thread = threading.Thread(target=self.posenet_worker)
        posenet_thread.start()
        
        mouse_timer = 0
        mouse_frame_count = 0
  
        is_ready = False
        
        last_frame = None
  
        while self.running:
            if not self.paused:
                current_time = time.time()
                delta_time = (current_time - last_frame) if last_frame else 0.0
                last_frame = current_time
                mouse_frame_count += 1
                
                if self.output_queue.empty() and is_ready:
                    self.mouse_controller.process(None, None, None, delta_time)
                else:
                    current = self.output_queue.get()
    
                    x, y, click = current['x'], current['y'], current['click']

                    self.mouse_controller.process(x, y, click, delta_time)
                    
                    is_ready = True
                    
                current_time = time.time()
                if current_time - mouse_timer > 2.0:
                    print(f"Mouse FPS: {mouse_frame_count / 2.0}")
                    mouse_frame_count = 0
                    mouse_timer = current_time
                    
            time.sleep(1 / FRAMERATE)
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        print(f"\n{'='*60}")
        print("Shutting down!")
        
        if self.output:
            try:
                self.output.Close()
                print("Output stream closed")
            except:
                pass
        
        if self.input:
            try:
                self.input.Close()
                print("Input stream closed")
            except:
                pass
        
        print(f"\nTotal clicks detected: {self.click_count}")
        print("Cleanup completed")
        print(f"\n{'='*60}")

# Initialize PoseNet globally
print("Initializing PoseNet...")
pose_net = poseNet("resnet18-hand")
print("PoseNet initialized successfully")

def main():
    """Main entry point"""
    # Parse command line arguments
    target_fps = 60  # Default target FPS
    
    
    # Create and run tracker
    tracker = FistHandTracker(target_fps=target_fps)
    tracker.run()


if __name__ == "__main__":
    main()