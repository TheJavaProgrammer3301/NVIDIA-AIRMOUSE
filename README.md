# NVIDIA AIRMOUSE
NVIDIA AIRMOUSE allows you to control your mouse when you don't feel like holding it or sitting down at your desk. Launch the program, then move your hand around to control the cursor. Make a fist to start clicking, and release the fist to stop.

## Behind the scenes
AIRMOUSE uses `PoseNet` for both movement control and gesture detection. We use `pyautogui` to move the cursor based on the input we get.

AIRMOUSE uses multithreading. PoseNet itself is not really capable of running in realtime (the fastest it can seem to get is 10 fps), so we decided to put PoseNet into its own special thread. That processing thread captures an image and processes it all its own pace, while the main thread (responsible for controlling the mouse) is free to update at frequencies of greater than 100 hz (though it's set to 30 hz by default to make the mouse move consistently).

We determine whether or not the user is attempting to click (and thus, whether they are making a fist gesture) by taking the average of the distances from the fingertips to either the center of the palm or the wrist, and testing whether it is within a certain threshold. The threshold is calculated dynamically based on the resolution of the screen.

To make clicking more stable, we require a configurable number of sequential frames detecting either the presence of a fist or the lack thereof before we change from clicking to not clicking, or vice versa. Whenever we stop clicking, we check if the length of the click (the number of sequential frames where we detected a fist, before it stopped) is less than or equal to our single click threshold (which is 2 times the click change threshold). If it is, then instead of starting/stopping a click using pyautogui.mouseDown or pyautogui.mouseUp, we active a single click through pyautogui.click, which makes it very easy to use ui elements that change behaviors based on how long you click on them.  

Every time we get an output from PoseNet, we look through all the keypoints provided to see if one matches the ID of the keypoint we are set to track. If so, we add that output (along with the outputs of the gesture detector) to a queue that we set up in the main thread. 

A simple smoothing algorithm is applied to the MouseController to prevent the controls from being too jumpy. Because of this smoothing algorithm, the main thread is able to update the mouse position even when it hasn't received any new input (since it can just keep moving the cursor to the last known input that we got).

PoseNet can also be a little spotty near the edges of the camera, and accommodate for this, we apply a multiplier (usually 2x) to the tracked keypoint position we found, relative to the center of the camera.

## Running the program
### Pip Dependencies
- pyautogui
- keyboard
### Pre-run
1. Make sure that the Jetson Nano is connected to a monitor.
	- AIRMOUSE works best when the ui is larger. For 4k monitors, you should change the UI scale to 2, by going to System Settings -> Displays -> Scale for menu and title bars
2. Make sure a camera is connected (preferably with a high field of view)
3. Setup permissions by running
	```bash
	xhost +SI:localuser:root
	```
### Actually running it
The parameters shown are defaults. 
**Press enter to stop the program.
Press space to pause/unpause.**
```bash
sudo python3 main.py --frequency=30 --multiplier=2 --keypoint=0 --click-threshold=3
```

### Examples
## Beautiful art made with NVIDIA AIRMOUSE
![art1.png](https://github.com/TheJavaProgrammer3301/NVIDIA-AIRMOUSE/blob/main/public/art1.png)
