import re
import csv

def parse_logs():
    # Dictionary to store start and end times for each frame
    frame_data = {}
    
    # Regular expressions to match the log patterns
    start_pattern = r'Started processing (\d+) at ([\d.]+)'
    end_pattern = r'Ended processing (\d+) at ([\d.]+)'
    
    try:
        with open('logs6.txt', 'r') as file:
            for line in file:
                line = line.strip()
                
                # Check for start pattern
                start_match = re.search(start_pattern, line)
                if start_match:
                    frame_id = int(start_match.group(1))
                    start_time = float(start_match.group(2))
                    
                    if frame_id not in frame_data:
                        frame_data[frame_id] = {}
                    frame_data[frame_id]['start'] = start_time
                    continue
                
                # Check for end pattern
                end_match = re.search(end_pattern, line)
                if end_match:
                    frame_id = int(end_match.group(1))
                    end_time = float(end_match.group(2))
                    
                    if frame_id not in frame_data:
                        frame_data[frame_id] = {}
                    frame_data[frame_id]['end'] = end_time
    
    except FileNotFoundError:
        print("Error: logs6.txt file not found")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Write to CSV
    try:
        with open('logs6.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Sort frames by ID for consistent output
            sorted_frames = sorted(frame_data.keys())
            
            if sorted_frames:
                # Write header row with frame numbers
                header = [''] + [f'frame {frame_id}' for frame_id in sorted_frames]
                writer.writerow(header)
                
                # Collect all start and end times for each frame
                start_times = []
                end_times = []
                
                for frame_id in sorted_frames:
                    data = frame_data[frame_id]
                    if 'start' in data and 'end' in data:
                        start_times.append(data['start'])
                        end_times.append(data['end'])
                    else:
                        start_times.append('')
                        end_times.append('')
                
                # Write start1 row
                writer.writerow(['start1'] + start_times)
                
                # Write start2 row (duplicate of start1)
                writer.writerow(['start2'] + start_times)
                
                # Write end1 row
                writer.writerow(['end1'] + end_times)
                
                # Write end2 row (duplicate of end1)
                writer.writerow(['end2'] + end_times)
        
        print(f"Successfully processed {len(sorted_frames)} frames and saved to logs6.csv")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    parse_logs()