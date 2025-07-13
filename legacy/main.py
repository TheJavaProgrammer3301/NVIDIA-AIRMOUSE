import os
import pyautogui
import numpy as np
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future
from queue import Queue, Full, Empty
from typing import Callable, Any, Optional
# from dataclasses import dataclass
from jetson_inference import depthNet, poseNet
from jetson_utils import videoSource, cudaDeviceSynchronize, videoOutput, Log, cudaToNumpy, cudaAllocMapped, cudaMemcpy
from depthnet_utils import depthBuffers

# Initialize networks globally (they're thread-safe for inference)
depth_net = depthNet("fcn-mobilenet")
pose_net = poseNet("resnet18-hand")

def draw_marker(x: float, y: float, rows=9, cols=16, char='X'):
    assert 0 <= x <= 1 and 0 <= y <= 1, "x and y must be in [0, 1]"

    row = int(y * (rows - 1))
    col = int(x * (cols - 1))

    for r in range(rows):
        line = ''
        for c in range(cols):
            line += char if (r == row and c == col) else ' '
        print(f"[ {line} ]")


# @dataclass
class ProcessingResult:
    img_input: Any
    poses: Any
    depth_field: Optional[Any] = None
    timestamp: float = 0.0
    task_id: int = 0
    processing_time: float = 0.0
    
    def __init__(self, img_input, poses, timestamp, task_id, processing_time, depth_field=None):
        self.img_input = img_input
        self.poses = poses
        self.depth_field = depth_field
        self.timestamp = timestamp
        self.task_id = task_id
        self.processing_time = processing_time

class MouseController:
    def __init__(self, screen_width=1920, screen_height=1080, camera_width=640, camera_height=480):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        self.smooth_factor = 0.3
        self.prev_x = screen_width // 2
        self.prev_y = screen_height // 2
        
        self.movement_threshold = 10
        
        # Properties for depthnet
        self.click_threshold = 0.15
        self.depth_history = deque(maxlen=15)
        self.baseline_depth = 0
        self.click_cooldown = 0
        self.min_click_frames = 3
        self.forward_motion_count = 0
        
        # Thread safety
        self.lock = threading.Lock()

    def update_mouse_position(self, finger_x, finger_y):
        with self.lock:
            screen_x = int((1.0 - finger_x / self.camera_width) * self.screen_width)
            screen_y = int((finger_y / self.camera_height) * self.screen_height)
            
            smooth_x = int(self.prev_x + (screen_x - self.prev_x) * self.smooth_factor)
            smooth_y = int(self.prev_y + (screen_y - self.prev_y) * self.smooth_factor)
            
            distance = ((smooth_x - self.prev_x) ** 2 + (smooth_y - self.prev_y) ** 2) ** 0.5
            if distance > self.movement_threshold:
                smooth_x = max(0, min(smooth_x, self.screen_width - 1))
                smooth_y = max(0, min(smooth_y, self.screen_height - 1))
                
                try:
                    pyautogui.moveTo(smooth_x, smooth_y)
                    self.prev_x = smooth_x
                    self.prev_y = smooth_y
                except pyautogui.FailSafeException:
                    print("Fail safe, please move your mouse away from the corners")
                    return False
        return True

    def process_depth_for_click(self, finger_x, finger_y, depth_field):
        with self.lock:
            if self.click_cooldown > 0:
                self.click_cooldown -= 1
                return False
            
            if depth_field is None:
                return False
                
            depth_numpy = cudaToNumpy(depth_field)
            x = int(max(0, min(finger_x, self.camera_width - 1)))
            y = int(max(0, min(finger_y, self.camera_height - 1)))
            
            region_size = 2
            y_start = max(0, y - region_size)
            y_end = min(self.camera_height, y + region_size + 1)
            x_start = max(0, x - region_size)
            x_end = min(self.camera_width, x + region_size + 1)
            
            depth_region = depth_numpy[y_start:y_end, x_start:x_end]
            current_depth = np.mean(depth_region)
            
            if self.baseline_depth == 0:
                self.baseline_depth = current_depth
                return False
            
            self.depth_history.append(current_depth)
            
            if len(self.depth_history) < 5:
                return False
            
            depth_change = (self.baseline_depth - current_depth) / self.baseline_depth
            
            if depth_change > self.click_threshold:
                self.forward_motion_count += 1
            else:
                if self.forward_motion_count >= self.min_click_frames:
                    self.forward_motion_count = 0
                    self.click_cooldown = 60
                    return True
                self.forward_motion_count = 0
            
            self.baseline_depth = 0.95 * self.baseline_depth + 0.05 * current_depth
            
            return False

    def center_calibration(self):
        with self.lock:
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            
            try:
                pyautogui.moveTo(center_x, center_y)
                self.prev_x = center_x
                self.prev_y = center_y
                print("Mouse cursor centered")
            except:
                print("Some error occurred with the centering calibration")

    def mouse_click(self):
        try:
            pyautogui.click()
            print("The mouse clicked")
        except:
            print("Some error occurred with the mouse click")

# Thread-local storage for depth buffers
thread_local = threading.local()

def get_thread_depth_buffer(shape, format):
    """Get or create a thread-local depth buffer"""
    if not hasattr(thread_local, 'depth_buffer') or thread_local.depth_buffer.shape != shape:
        thread_local.depth_buffer = cudaAllocMapped(width=shape[1], height=shape[0], format=format)
    return thread_local.depth_buffer

def process_frame_with_networks(img_input, task_id, timeout=0.1):
    """Process a single frame through both pose and depth networks"""
    start_time = time.time()
    
    try:
        print(f"Started processing {task_id} at {time.perf_counter()}")
  
        # Create a copy of the input for depth processing
        # img_copy = cudaAllocMapped(width=img_input.width, height=img_input.height, format=img_input.format)
        # cudaMemcpy(img_copy, img_input)
        
        # Get thread-local depth buffer
        depth_buffer = get_thread_depth_buffer(img_input.shape, img_input.format)
        
        # Process pose detection
        poses = pose_net.Process(img_input)
        
        # Process depth estimation
        depth_net.Process(img_input, depth_buffer, "viridis-inverted", "linear")
        
        processing_time = time.time() - start_time
        
        print(f"Ended processing {task_id} at {time.perf_counter()}")
        
        # Check if we've exceeded timeout
        if processing_time > timeout:
            print(f"Frame {task_id} processing took too long: {processing_time:.3f}s")
            return None
        
        return ProcessingResult(
            img_input=img_input,
            poses=poses,
            depth_field=depth_buffer,
            timestamp=start_time,
            task_id=task_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"Error processing frame {task_id}: {e}")
        return None

class TimeoutThreadPoolProcessor:
    def __init__(self, pool_size: int, handler: Callable[[ProcessingResult], None], 
                 queue_size: int = 10, frame_timeout: float = 0.1, max_pending_frames: int = 5):
        self.pool_size = pool_size
        self.handler = handler
        self.queue_size = queue_size
        self.frame_timeout = frame_timeout
        self.max_pending_frames = max_pending_frames
        
        self.executor = ThreadPoolExecutor(max_workers=pool_size)
        self.input_queue = Queue(maxsize=queue_size)
        self.pending_futures = {}
        self.results = {}
        
        self.task_counter = 0
        self.expected_task_id = 0
        self.lock = threading.Lock()
        
        self.running = True
        self.stats = {
            'processed': 0,
            'dropped': 0,
            'timed_out': 0
        }
        
        # Start result handler thread
        self.result_thread = threading.Thread(target=self._result_handler)
        self.result_thread.daemon = True
        self.result_thread.start()

    def add_frame(self, img_input) -> bool:
        """Add a frame for processing. Returns False if dropped."""
        with self.lock:
            # Check if we have too many pending frames
            if len(self.pending_futures) >= self.max_pending_frames:
                self.stats['dropped'] += 1
                print(f"Dropping frame - too many pending ({len(self.pending_futures)})")
                return False
            
            task_id = self.task_counter
            self.task_counter += 1
        
        try:
            # Submit task to thread pool
            future = self.executor.submit(process_frame_with_networks, img_input, task_id, self.frame_timeout)
            
            with self.lock:
                self.pending_futures[task_id] = (future, time.time())
            
            return True
            
        except Exception as e:
            print(f"Failed to submit frame {task_id}: {e}")
            self.stats['dropped'] += 1
            return False

    def _result_handler(self):
        """Handle completed results and maintain order"""
        while self.running:
            current_time = time.time()
            completed_tasks = []
            
            with self.lock:
                # Check for completed or timed-out futures
                for task_id, (future, submit_time) in list(self.pending_futures.items()):
                    if future.done():
                        try:
                            result = future.result(timeout=0)
                            if result:
                                self.results[task_id] = result
                                self.stats['processed'] += 1
                                
                                print(f"Task {task_id} finished successfully")
                            else:
                                self.stats['timed_out'] += 1
                        except Exception as e:
                            print(f"Task {task_id} failed: {e}")
                            self.stats['dropped'] += 1
                        
                        completed_tasks.append(task_id)
                    
                    elif current_time - submit_time > self.frame_timeout * 2:
                        # Force timeout after 2x the frame timeout
                        future.cancel()
                        self.stats['timed_out'] += 1
                        completed_tasks.append(task_id)
                        print(f"Force timeout frame {task_id}")
                
                # Remove completed tasks
                for task_id in completed_tasks:
                    del self.pending_futures[task_id]
                
                # Deliver results in order
                while self.expected_task_id in self.results:
                    result = self.results.pop(self.expected_task_id)
                    self.expected_task_id += 1
                    
                    # Call handler in a separate thread to avoid blocking
                    threading.Thread(target=self.handler, args=(result,), daemon=True).start()
                
                # Skip missing frames if we're too far behind
                if self.results and min(self.results.keys()) > self.expected_task_id + 10:
                    skip_to = min(self.results.keys())
                    print(f"Skipping frames {self.expected_task_id} to {skip_to-1}")
                    self.expected_task_id = skip_to
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting

    def get_stats(self):
        with self.lock:
            return self.stats.copy()

    def shutdown(self):
        self.running = False
        self.executor.shutdown(wait=True)
        self.result_thread.join()

class ParallelHandTracker:
    def __init__(self, num_workers=8, frame_timeout=0.1):
        # Get screen resolution
        try:
            self.screen_width, self.screen_height = pyautogui.size()
            print(f'Screen resolution: {self.screen_width}x{self.screen_height}')
        except:
            self.screen_width, self.screen_height = 1920, 1080
            print(f'Revert to default resolution: {self.screen_width}x{self.screen_height}')
        
        # Initialize components
        self.input = videoSource("/dev/video0")
        self.output = videoOutput("webrtc://@:8554/output")
        self.mouse_controller = MouseController(self.screen_width, self.screen_height)
        
        self.camera_width = self.input.GetWidth()
        self.camera_height = self.input.GetHeight()
        
        # Create processor with timeout capabilities
        self.processor = TimeoutThreadPoolProcessor(
            pool_size=num_workers,
            handler=self.handle_result,
            queue_size=num_workers * 2,
            frame_timeout=frame_timeout,
            max_pending_frames=num_workers * 2
        )
        
        self.POINT_FINGER_TIP = 8
        self.last_stats_time = time.time()
        
        self._warmup_network()
        
    def _warmup_network(self):
        """Warm up DepthNet to avoid slow first frame"""
        print("Warming up DepthNet...")
        try:
            # Create dummy image
            dummy_img = cudaAllocMapped(
                width=self.camera_width,
                height=self.camera_height,
                format="rgb8"
            )
            
            buffers = cudaAllocMapped(width=dummy_img.shape[1], height=dummy_img.shape[0], format=dummy_img.format)
            
            # # Allocate depth buffer
            # if not self.buffers.depth:
            #     self.buffers.Alloc(dummy_img.shape, dummy_img.format)
            
            # Process dummy frame
            start = time.time()
            depth_net.Process(dummy_img, buffers, "viridis-inverted", "linear")
            warmup_time = time.time() - start
            
            print(f"DepthNet warmup completed in {warmup_time*1000:.1f}ms")
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")

    def handle_result(self, result: ProcessingResult):
        """Handle processed frame results"""
        if not result:
            return
        
        # Find finger tip
        finger_tip = None
        for pose in result.poses:
            for keypoint in pose.Keypoints:
                if keypoint.ID == self.POINT_FINGER_TIP:
                    finger_tip = keypoint
                    break
            if finger_tip:
                break
        
        if finger_tip:
            # Update mouse position
            # self.mouse_controller.update_mouse_position(finger_tip.x, finger_tip.y)
            
            draw_marker(1 - (finger_tip.x / self.camera_width), finger_tip.y / self.camera_height)
            
            # # Check for click using depth
            # if result.depth_field and self.mouse_controller.process_depth_for_click(
            #         finger_tip.x, finger_tip.y, result.depth_field):
            #     self.mouse_controller.mouse_click()
            
            
        # Output the processed frame
        if self.output:
            self.output.Render(result.img_input)

    def run(self):
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                # Capture frame
                img_input = self.input.Capture(format='rgb8', timeout=1000)
                if img_input is None:
                    continue
                
                # Add frame to processing queue
                self.processor.add_frame(img_input)
                
                frame_count += 1
                
                # Print stats every 5 seconds
                if time.time() - self.last_stats_time > 5.0:
                    stats = self.processor.get_stats()
                    print(f"\nStats - Processed: {stats['processed']}, "
                          f"Dropped: {stats['dropped']}, "
                          f"Timed out: {stats['timed_out']}, "
                          f"FPS: {frame_count/5:.1f}")
                    frame_count = 0
                    self.last_stats_time = time.time()
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            self.processor.shutdown()
            if self.output:
                self.output.Close()
            if self.input:
                self.input.Close()

# Main execution
if __name__ == "__main__":
    # Configuration
    num_workers = 4
    frame_timeout = 35  # 100ms timeout per frame
    
    print(f"Starting with {num_workers} worker threads, {frame_timeout}s timeout per frame...")
    
    tracker = ParallelHandTracker(num_workers=num_workers, frame_timeout=frame_timeout)
    tracker.run()