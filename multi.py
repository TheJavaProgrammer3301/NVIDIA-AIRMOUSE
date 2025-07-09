import time
import threading
import queue
import numpy as np
import sys
import cv2
from jetson_inference import depthNet, poseNet
from jetson_utils import videoSource, cudaAllocMapped, cudaToNumpy, cudaFromNumpy, cudaMemcpy

# Constants
NUM_WORKERS = 4

# Global lock to prevent concurrent initialization
init_lock = threading.Lock()

# Central array for processed frames with thread-safe access
processed_frames = []
frames_lock = threading.Lock()

def draw_marker(x: float, y: float, rows=9, cols=16, char='X'):
    assert 0 <= x <= 1 and 0 <= y <= 1, "x and y must be in [0, 1]"

    row = int(y * (rows - 1))
    col = int(x * (cols - 1))

    for r in range(rows):
        line = ''
        for c in range(cols):
            line += char if (r == row and c == col) else ' '
        print(f"[ {line} ]")

class ProcessedFrame:
    def __init__(self, frame, poses, finger_tip, depth_value, completion_time, thread_id):
        self.frame = frame
        self.poses = poses
        self.finger_tip = finger_tip
        self.depth_value = depth_value
        self.completion_time = completion_time
        self.thread_id = thread_id

class WorkerThread:
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.pose_net = None
        self.frame_queue = queue.Queue(maxsize=1)  # Only hold one frame at a time
        self.running = True
        self.ready_event = threading.Event()
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True  # Make thread a daemon
        self.thread.start()
    
    def _worker_loop(self):
        """Main worker thread loop"""
        # Initialize networks
        with init_lock:
            # Initialize PoseNet
            start_time_posenet = time.perf_counter()
            print(f"!!! THREAD {self.thread_id} STARTING POSENET", start_time_posenet)
            self.pose_net = poseNet("resnet18-hand")
            end_time_posenet = time.perf_counter()
            print(f"!!! THREAD {self.thread_id} DONE STARTING POSENET", end_time_posenet, "Elapsed time:", end_time_posenet - start_time_posenet)
        
        # Signal that this thread is ready
        self.ready_event.set()
        
        # Main processing loop
        while self.running:
            try:
                # Wait for a frame with timeout
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:  # Shutdown signal
                    break
                
                # Process frame with pose detection (applies overlay in place)
                start_pose = time.perf_counter()
                poses = self.pose_net.Process(frame, overlay="keypoints")
                end_pose = time.perf_counter()
                pose_time = end_pose - start_pose
                
                # Process finger tip position
                finger_tip = None
                
                for pose in poses:
                    for keypoint in pose.Keypoints:
                        if keypoint.ID == 8:
                            finger_tip = keypoint
                            break
                    if finger_tip:
                        break
                    
                x, y = (finger_tip.x, finger_tip.y) if finger_tip else (None, None)
                # depth_value = self.depth_numpy[y, x] if finger_tip else None
                depth_value = None
                
                completion_time = time.perf_counter()
                
                # Create processed frame object
                processed_frame = ProcessedFrame(
                    frame=frame,
                    poses=poses,
                    finger_tip=(x, y),
                    depth_value=depth_value,
                    completion_time=completion_time,
                    thread_id=self.thread_id
                )
                
                # Add to central array with thread-safe access
                with frames_lock:
                    processed_frames.append(processed_frame)
                    # Sort by completion time to maintain chronological order
                    processed_frames.sort(key=lambda pf: pf.completion_time)
                    
                    # Optional: Limit array size to prevent memory issues
                    # if len(processed_frames) > 100:  # Keep last 100 frames
                    #     processed_frames.pop(0)
                
                print(f"{self.thread_id} took {pose_time:.4f} seconds to pose. Finger tip at ({x}, {y})")
                if finger_tip:
                    draw_marker(x / frame.width, y / frame.height, rows=20, cols=35, char='X')
                
                with open("output.txt", "a") as f:
                    f.write(f"{completion_time}: ({x}, {y})\n")
                    f.close()
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, continue waiting
    
    def is_free(self):
        """Check if this worker is free to accept a new frame"""
        return self.frame_queue.empty()
    
    def process_frame(self, frame):
        """Submit a frame for processing (non-blocking)"""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            return False
    
    def shutdown(self):
        """Shutdown the worker thread"""
        self.running = False
        self.frame_queue.put(None)  # Signal shutdown
        self.thread.join()

class MainProcessor:
    def __init__(self, num_workers=NUM_WORKERS, video_source_path="/dev/video0"):
        self.num_workers = num_workers
        self.video_source_path = video_source_path
        self.workers = []
        self.video_source = None
        self.running = False
        self.depth_net = None
        self.depth_numpy = None
        
    def initialize(self):
        """Initialize the processor with worker threads and video source"""
        print(f"Initializing MainProcessor with {self.num_workers} workers...")
        
        # Create worker threads
        self.workers = [WorkerThread(i) for i in range(self.num_workers)]
        
        # Wait for all threads to complete initialization
        for worker in self.workers:
            worker.ready_event.wait()
        
        print("!!! ALL THREADS ARE READY - INITIALIZATION COMPLETE")
        
        # Create video source
        self.video_source = videoSource(self.video_source_path)
        
        # Initialize DepthNet
        start_time_depthnet = time.perf_counter()
        print(f"!!! STARTING DEPTHNET", start_time_depthnet)
        self.depth_net = depthNet("fcn-mobilenet")
        self.depth_numpy = cudaToNumpy(self.depth_net.GetDepthField())
        end_time_depthnet = time.perf_counter()
        print(f"!!! DONE STARTING DEPTHNET", end_time_depthnet, "Elapsed time:", end_time_depthnet - start_time_depthnet)
        
        # Create dummy frame and test depthNet
        print(f"!!! TESTING DEPTHNET WITH DUMMY FRAME")
        dummy_frame = cudaAllocMapped(width=640, height=480, format='rgb8')
        dummy_depth = self.depth_net.Process(dummy_frame)
        print(f"!!! DEPTHNET DUMMY FRAME TEST COMPLETE")
        
        print(f"!!! INITIALIZATION COMPLETE")
        
        return True
    
    def do_depthnet(self, frame):
        """Process frame with depth estimation"""
        self.depth_net.Process(frame)
    
    def process_frame(self, frame):
        """Process a single frame by dispatching to available worker"""
        if not frame:
            return False
            
        # Create a copy of the frame for processing
        frame_copy = cudaAllocMapped(width=frame.width, 
                                   height=frame.height, 
                                   format=frame.format)
        # Copy the data
        cudaMemcpy(frame_copy, frame)
        
        # Find a free worker thread
        free_worker = None
        for worker in self.workers:
            if worker.is_free():
                free_worker = worker
                break
        
        try:
            if free_worker:
                # Send frame to worker thread
                if free_worker.process_frame(frame_copy):
                    print(f"Frame queued to worker {free_worker.thread_id}")
                    return True
                else:
                    print("Frame dropped - worker queue full")
                    return False
            else:
                print("Frame dropped - no free worker threads")
                return False
        except Exception as e:
            print(f"Error processing frame: {e}")
            return False
    
    def run(self):
        """Main processing loop"""
        if not self.initialize():
            print("Failed to initialize processor")
            return
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.perf_counter()
                frame = self.video_source.Capture()
                end_time = time.perf_counter()
                
                if frame is not None:
                    capture_time = end_time - start_time
                    fps = 1.0 / capture_time
                    print(f"Frame capture FPS: {fps:.2f}")
                    
                    # Process the frame
                    success = self.process_frame(frame)
                    if not success:
                        print("Failed to process frame")
                else:
                    print("Failed to capture frame")
        
        except KeyboardInterrupt:
            print("Shutting down...")
            self.shutdown()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the processor and cleanup resources"""
        print("Shutting down MainProcessor...")
        self.running = False
        
        # Render processed frames to video
        print("Rendering processed frames to video...")
        render_frames_to_video("pose_detection_output.mp4", base_fps=30)
        
        # Save frame metadata
        save_frame_metadata("pose_detection_metadata.txt")
        
        print("MainProcessor cleanup complete.")
    
    def get_worker_count(self):
        """Get the number of worker threads"""
        return len(self.workers)
    
    def get_worker_status(self):
        """Get status of all worker threads"""
        status = []
        for worker in self.workers:
            status.append({
                'thread_id': worker.thread_id,
                'is_free': worker.is_free(),
                'is_running': worker.running
            })
        return status

def render_frames_to_video(output_filename="output_video.mp4", base_fps=30):
    """Render all processed frames to a video file with timing based on actual processing times"""
    with frames_lock:
        if not processed_frames:
            print("No processed frames to render")
            return
        
        frames_to_render = processed_frames.copy()
    
    print(f"Rendering {len(frames_to_render)} frames to video: {output_filename}")
    
    if len(frames_to_render) < 2:
        print("Need at least 2 frames to calculate timing")
        return
    
    # Get frame dimensions from the first frame
    first_frame = frames_to_render[0].frame
    frame_numpy = cudaToNumpy(first_frame)
    height, width = frame_numpy.shape[:2]
    
    # Calculate timing information
    start_time = frames_to_render[0].completion_time
    end_time = frames_to_render[-1].completion_time
    total_duration = end_time - start_time
    
    print(f"Total processing duration: {total_duration:.2f} seconds")
    print(f"Average processing rate: {len(frames_to_render)/total_duration:.2f} fps")
    
    # Initialize video writer with base fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, base_fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_filename}")
        return
    
    try:
        current_video_time = 0.0
        frame_duration = 1.0 / base_fps  # Duration of each video frame in seconds
        
        for i, processed_frame in enumerate(frames_to_render):
            # Calculate when this frame should appear in the video based on processing time
            frame_processing_time = processed_frame.completion_time - start_time
            
            # Convert CUDA frame to numpy array
            frame_numpy = cudaToNumpy(processed_frame.frame)
            
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2BGR)
            
            # Add frame information overlay with timing info
            info_text = f"Frame {i+1}/{len(frames_to_render)} | Thread {processed_frame.thread_id}"
            if processed_frame.finger_tip[0] is not None:
                info_text += f" | Finger: ({processed_frame.finger_tip[0]:.0f}, {processed_frame.finger_tip[1]:.0f})"
            
            timing_text = f"Time: {frame_processing_time:.2f}s"
            
            cv2.putText(frame_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, timing_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Write frames to fill the time gap until this frame should appear
            while current_video_time < frame_processing_time:
                video_writer.write(frame_bgr)
                current_video_time += frame_duration
            
            # If we have a gap between frames, duplicate the current frame
            if i < len(frames_to_render) - 1:
                next_frame_time = frames_to_render[i + 1].completion_time - start_time
                while current_video_time < next_frame_time:
                    video_writer.write(frame_bgr)
                    current_video_time += frame_duration
            
            if (i + 1) % 10 == 0:
                print(f"Rendered {i + 1}/{len(frames_to_render)} frames... (video time: {current_video_time:.2f}s)")
        
        print(f"Successfully rendered video: {output_filename}")
        print(f"Video duration: {current_video_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error rendering video: {e}")
    finally:
        video_writer.release()

def save_frame_metadata(filename="frame_metadata.txt"):
    """Save frame processing metadata to a text file"""
    with frames_lock:
        if not processed_frames:
            print("No frame metadata to save")
            return
        
        frames_to_save = processed_frames.copy()
    
    try:
        with open(filename, 'w') as f:
            f.write("Frame Index,Thread ID,Completion Time,Finger Tip X,Finger Tip Y,Depth Value\n")
            for i, pf in enumerate(frames_to_save):
                f.write(f"{i},{pf.thread_id},{pf.completion_time:.6f},{pf.finger_tip[0]},{pf.finger_tip[1]},{pf.depth_value}\n")
        
        print(f"Frame metadata saved to: {filename}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

def get_latest_processed_frames(count=10):
    """Get the latest processed frames (sorted by completion time)"""
    with frames_lock:
        return processed_frames[-count:] if len(processed_frames) >= count else processed_frames.copy()

def get_all_processed_frames():
    """Get all processed frames (sorted by completion time)"""
    with frames_lock:
        return processed_frames.copy()

def main():
    """Main function that creates and runs the processor"""
    # Create and run the main processor
    processor = MainProcessor(num_workers=NUM_WORKERS, video_source_path="/dev/video0")
    processor.run()

if __name__ == "__main__":
    main()