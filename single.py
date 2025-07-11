import os
import sys
import pyautogui
import numpy as np
import time
import queue
import threading
import keyboard
from collections import deque
from typing import Optional, Tuple
# from dataclasses import dataclass
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, cudaAllocMapped

# Initialize PoseNet globally
print("Initializing PoseNet...")
pose_net = poseNet("resnet18-hand")
print("PoseNet initialized successfully")

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 1 / 30
pyautogui.DARWIN_CATCH_UP_TIME = 0

# @dataclass
class ProcessingResult:
    poses: any
    finger_tip: Optional[any] = None
    processing_time: float = 0.0
    frame_id: int = 0
    click_detected: bool = False
    gesture_type: str = ""
    
    def __init__(self, poses, frame_id):
        self.poses = poses
        self.finger_tip = None
        self.processing_time = None
        self.frame_id = frame_id
        self.click_detected = False
        self.gesture_type = ""


class GestureClickDetector:
    """Detect pinch and fist gestures for clicking"""
    
    def __init__(self, click_mode="pinch"):
        # Hand keypoint IDs
        self.WRIST = 0
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_MCP = 5
        self.INDEX_PIP = 6
        self.INDEX_DIP = 7
        self.INDEX_TIP = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_PIP = 10
        self.MIDDLE_DIP = 11
        self.MIDDLE_TIP = 12
        self.RING_MCP = 13
        self.RING_PIP = 14
        self.RING_DIP = 15
        self.RING_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20
        
        # Click detection parameters
        self.click_mode = click_mode  # "pinch", "fist", or "both"
        # self.click_cooldown = 0
        # self.click_cooldown_frames = 15
        
        # Gesture thresholds
        self.pinch_threshold = 80  # pixels
        self.fist_threshold = 80   # pixels
        
        # Gesture history for stability
        self.gesture_history = deque(maxlen=5)
        self.last_gesture_state = False
        
        # Calibration
        self.hand_scale = 1.0
        self.calibration_samples = deque(maxlen=30)
        
        print(f"Gesture detector initialized with mode: {click_mode}")
    
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
            if kp.ID == self.WRIST:
                wrist = kp
            elif kp.ID == self.MIDDLE_TIP:
                middle_tip = kp
        
        if wrist and middle_tip:
            hand_length = self.calculate_distance(wrist, middle_tip)
            self.calibration_samples.append(hand_length)
            
            if len(self.calibration_samples) >= 20:
                avg_hand_length = np.mean(self.calibration_samples)
                # Scale thresholds based on hand size (normalized to ~150 pixel hand)
                self.hand_scale = avg_hand_length / 150.0
                self.pinch_threshold = 40 * self.hand_scale
                self.fist_threshold = 80 * self.hand_scale
    
    def detect_pinch(self, keypoints) -> bool:
        """Detect pinch gesture (thumb tip close to index tip)"""
        thumb_tip = None
        index_tip = None
        
        for kp in keypoints:
            if kp.ID == self.THUMB_TIP:
                thumb_tip = kp
            elif kp.ID == self.INDEX_TIP:
                index_tip = kp
        
        if thumb_tip and index_tip:
            distance = self.calculate_distance(thumb_tip, index_tip)
            # print(f"pinch: {distance / self.pinch_threshold}")
            return distance < self.pinch_threshold
        return False
    
    def detect_fist(self, keypoints) -> bool:
        """Detect closed fist (all fingertips close to palm center)"""
        wrist = None
        fingertips = []
        palm_center = None
        
        # Get key points
        for kp in keypoints:
            if kp.ID == self.WRIST:
                wrist = kp
            elif kp.ID in [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]:
                fingertips.append(kp)
            elif kp.ID == self.MIDDLE_MCP:  # Use middle MCP as palm center reference
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
    
    def detect_click(self, pose) -> Tuple[bool, str]:
        """Main click detection method"""
        # if self.click_cooldown > 0:
        #     self.click_cooldown -= 1
        #     return False, ""
        
        if not pose or not pose.Keypoints:
            return False, ""
        
        # Auto-calibrate based on hand size
        self.auto_calibrate_hand_size(pose.Keypoints)
        
        # Detect gestures
        pinch_detected = False
        fist_detected = False
        gesture_type = ""
        
        if self.click_mode in ["pinch", "both"]:
            pinch_detected = self.detect_pinch(pose.Keypoints)
            if pinch_detected:
                gesture_type = "pinch"
        
        if self.click_mode in ["fist", "both"]:
            fist_detected = self.detect_fist(pose.Keypoints)
            if fist_detected:
                gesture_type = "fist"
        
        gesture_detected = pinch_detected or fist_detected
        
        # Add to history for stability
        self.gesture_history.append(gesture_detected)
        
        # Require consistent gesture for a few frames
        if len(self.gesture_history) >= 3:
            recent_gestures = list(self.gesture_history)[-3:]
            
            # Detect gesture onset (transition from no gesture to gesture)
            if not self.last_gesture_state and all(recent_gestures):
                self.last_gesture_state = True
                # self.click_cooldown = self.click_cooldown_frames
                return True, gesture_type
            elif self.last_gesture_state and not any(recent_gestures[-2:]):
                self.last_gesture_state = False
        
        return False, ""


class MouseController:
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, 
                 camera_width: int = 640, camera_height: int = 480,
                 click_mode: str = "pinch"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Mouse movement parameters
        self.smooth_factor = 0.4
        self.prev_x = screen_width // 2
        self.prev_y = screen_height // 2
        self.movement_threshold = 5
        
        self.center_x = camera_width // 2
        self.center_y = camera_height // 2
        
        self.finger_x = self.prev_x
        self.finger_y = self.prev_y
        
        # Center the mouse on initialization
        self.center_mouse()
        
        self.POSITION_MULTIPLIER = 1.5
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
    
    def update_mouse_position(self, finger_x: float, finger_y: float) -> bool:
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
    
    def set_clicking(self, clicking: bool, gesture_type: str = ""):
        """Execute mouse click"""
        try:
            clicking_changed = clicking != self.clicking
            
            if clicking_changed:
                if clicking:
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()
                
				self.clicking = clicking
    
            	print(f"CLICK: {clicking} ({gesture_type})")
        except Exception as e:
            print(f"Click failed: {e}")
            
    def process(self, finger_x: float, finger_y: float, clicked: bool) -> bool:
        if finger_x is not None:
            self.finger_x = max(0, min(((finger_x - self.center_x) * self.POSITION_MULTIPLIER) + self.center_x, self.camera_width - 1))
            self.finger_y = max(0, min(((finger_y - self.center_y) * self.POSITION_MULTIPLIER) + self.center_y, self.camera_height - 1))
            
            print(f"Finger position: ({self.finger_x}, {self.finger_y}); transformed from ({finger_x}, {finger_y})")
            
        self.update_mouse_position(self.finger_x, self.finger_y)
        
        self.set_clicking(clicked)

class PinchFistHandTracker:
    def __init__(self, target_fps: int = 60, click_mode: str = "pinch"):
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
            self.camera_width, self.camera_height,
            click_mode
        )
        
        # Click detection
        self.gesture_detector = GestureClickDetector(click_mode)
        
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
        self.POINT_FINGER_TIP = 8
        self.last_finger_pos = None
        self.finger_lost_frames = 0
        self.max_finger_lost_frames = 10
        
        # Statistics
        self.click_count = 0
        self.click_mode = click_mode
        
        # Warm up the network
        self._warmup_network()
        
        self.running = False
  
        self.output_queue = queue.Queue()
        
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
                    if keypoint.ID == self.POINT_FINGER_TIP:
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
            click_detected, gesture_type = self.check_for_click(primary_pose)
            result.click_detected = click_detected
            result.gesture_type = gesture_type
        
        result.processing_time = time.time() - start_time
        return result
    
    def check_for_click(self, pose) -> Tuple[bool, str]:
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
                  f"Clicks: {self.click_count} | Mode: {self.click_mode}")
    
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
                    'click': result.click_detected
                })
                
                # # Handle click
                # if result.click_detected:
                #     self.mouse_controller.perform_click(result.gesture_type)
                #     self.click_count += 1
                
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
        
        keyboard.on_press_key("space", lambda e: self.stop_program())
        
        # Print instructions
        print(f"\n{'='*60}")
        print(f"Starting PoseNet hand tracker (Target FPS: {self.target_fps})")
        print(f"Click detection mode: {self.click_mode.upper()}")
        print(f"{'='*60}")
        
        if self.click_mode == "pinch":
            print("GESTURE: Pinch your thumb and index finger together to click")
        elif self.click_mode == "fist":
            print("GESTURE: Make a fist to click")
        elif self.click_mode == "both":
            print("GESTURES: Pinch OR make a fist to click")
        
        print("\nMove your index finger to control the mouse")
        print("Press Ctrl+C to exit\n")
        
        self.mouse_controller.process(None, None, False)
        
        posenet_thread = threading.Thread(target=self.posenet_worker)
        posenet_thread.start()
        
        mouse_timer = 0
        mouse_frame_count = 0
  
        is_ready = False
  
        while self.running:
            if self.output_queue.empty() and is_ready:
                self.mouse_controller.process(None, None, False)
            else:
                current = self.output_queue.get()
   
                x, y, click = current['x'], current['y'], current['click']

                self.mouse_controller.process(x, y, click)
                
                is_ready = True
                
            mouse_frame_count += 1
                
            # time.sleep(1.0 / 120)
            
            current_time = time.time()
            if current_time - mouse_timer > 2.0:
                print(f"Mouse FPS: {mouse_frame_count / 2.0}")
                mouse_frame_count = 0
                mouse_timer = current_time
        
        # try:
        #     while self.running:
        #         # Capture frame
        #         img_input = self.input.Capture(format='rgb8', timeout=1000)
        #         if img_input is None:
        #             print("Failed to capture frame")
        #             continue
                
        #         # Check if we should skip this frame
        #         if self.should_skip_frame():
        #             continue
                
        #         # Process the frame
        #         result = self.process_frame(img_input)
        #         self.processing_times.append(result.processing_time)
                
        #         # Update mouse position if finger is detected
        #         x, y = (None, None) if result.finger_tip is None else (result.finger_tip.x, result.finger_tip.y)
                
        #         self.mouse_controller.process(
        #             x,
        #             y,
        #             result.click_detected
        #         )
                
        #         # # Handle click
        #         # if result.click_detected:
        #         #     self.mouse_controller.perform_click(result.gesture_type)
        #         #     self.click_count += 1
                
        #         # Render output if available
        #         if self.output:
        #             self.output.Render(img_input)
                
        #         # Update frame counter
        #         self.frame_count += 1
                
        #         # Print statistics every 2 seconds
        #         current_time = time.time()
        #         if current_time - self.fps_timer > 2.0:
        #             self.print_stats()
        #             self.frame_count = 0
        #             self.fps_timer = current_time
        
        # except KeyboardInterrupt:
        #     print("\n\nShutting down gracefully...")
        # except Exception as e:
        #     print(f"\nError in main loop: {e}")
        #     import traceback
        #     traceback.print_exc()
        # finally:
        #     self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
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


def main():
    """Main entry point"""
    # Parse command line arguments
    click_mode = "both"  #["pinch", "fist", "both"]
    target_fps = 60  # Default target FPS
    
    
    # Create and run tracker
    tracker = PinchFistHandTracker(target_fps=target_fps, click_mode=click_mode)
    tracker.run()


if __name__ == "__main__":
    main()