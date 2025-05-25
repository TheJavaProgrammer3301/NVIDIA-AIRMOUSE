from jetson_inference import depthNet, poseNet
from jetson_utils import videoSource, cudaDeviceSynchronize, Log
from depthnet_utils import depthBuffers

depth_net = depthNet("fcn-mobilenet")
pose_net = poseNet("resnet18-hand")

buffers = depthBuffers()

input = videoSource("/dev/video0")

while True:
	img_input = input.Capture()

	if img_input is None:
		continue

	buffers.Alloc(img_input.shape, img_input.format)

	depth_net.Process(img_input, buffers.depth, "viridis-inverted", "linear")
	poses = pose_net.Process(img_input)

	for pose in poses:
		print(pose)

	cudaDeviceSynchronize()

	depth_net.PrintProfilerTimes()

	if not input.IsStreaming():
		break