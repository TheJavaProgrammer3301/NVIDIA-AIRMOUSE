import os
import sys
import pyautogui
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, cudaAllocMapped

# Initialize network globally

pose_net = poseNet("resnet18-hand")


class ProcessingResult:
    poses: any
    finger_tip: Optional[any] = None
    processing_time: float = 0.0
    frame_id: int = 0

class MouseController:
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, 
                 camera_width: int = 640, camera_height: int = 480):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Mouse movement parameters
        self.smooth_factor = 0.4
        self.prev_x = screen_width // 2
        self.prev_y = screen_height // 2
        self.movement_threshold = 5
        
        # Center the mouse on initialization
        self.center_mouse()
    
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
        if distance > self.movement_threshold:
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
        
        return True

class PoseNetHandTracker:
    def __init__(self, target_fps: int = 60):
        """Initialize the hand tracker with PoseNet only"""
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
        self.POINT_FINGER_TIP = 8  # Index finger tip keypoint ID
        self.last_finger_pos = None
        self.finger_lost_frames = 0
        self.max_finger_lost_frames = 10
        
        # Warm up the network with a dummy frame
        self._warmup_network()
        
        self.running = False
    
    def _warmup_network(self):
        """Warm up the PoseNet to avoid slow first frame"""
        print("Dummy frame warming up PoseNet...")
        try:
            # Create a dummy image
            dummy_img = cudaAllocMapped(
                width=self.camera_width, 
                height=self.camera_height, 
                format="rgb8"
            )
            

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
        
        # Find index finger tip
        finger_found = False
        if result.poses:
            for pose in result.poses:
                if pose.Keypoints:
                    for keypoint in pose.Keypoints:
                        if keypoint.ID == self.POINT_FINGER_TIP:
                            result.finger_tip = keypoint
                            self.last_finger_pos = (keypoint.x, keypoint.y)
                            self.finger_lost_frames = 0
                            finger_found = True
                            break
                if finger_found:
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
        
        result.processing_time = time.time() - start_time
        return result
    
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
                  f"Skips: {self.skip_counter}")
    
    def run(self):
        """Main processing loop"""
        self.running = True
        print(f"\nStarting PoseNet hand tracker (Target FPS: {self.target_fps})")
        print("Move your index finger to control the mouse")
        print("Press Ctrl+C to exit\n")
        
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
                if result.finger_tip:
                    self.mouse_controller.update_mouse_position(
                        result.finger_tip.x,
                        result.finger_tip.y
                    )
                
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
        finally:
            self.cleanup()
    
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
        
        print("Cleanup completed")

def main():
    """Main entry point"""
    target_fps = 30 
    

    tracker = PoseNetHandTracker(target_fps=target_fps)
    tracker.run()

if __name__ == "__main__":
    main()