from django.conf import settings

import os

import numpy as np
import cv2



# Initial settings 
prototxt_ = "MobileNetSSD_deploy.prototxt.txt"
model_ = "MobileNetSSD_deploy.caffemodel"

# Class labels taken from PASCAL VOC dataset
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Project paths
prototxt = os.path.join(settings.RESOURCE_ROOT, prototxt_)
model = os.path.join(settings.RESOURCE_ROOT, model_)

## Functions #
# Detection function
def detection(image, networkModel=prototxt, trainedModel=model):
	# Image decoding to original format
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# Actual processing
	# Network  architecture loading
	net = cv2.dnn.readNetFromCaffe(networkModel, trainedModel)

	# Resize image to feed the network
	#dim = image.shape
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# Detection and prediction
	net.setInput(blob)
	detections = net.forward()

	succeeded = 0
	ini_x, ini_y = 0, 0

	object_detected = {}
	# Loop over the detections
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			# Get the index of label
			id_label = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
			(ini_x, ini_y, end_x, end_y) = box.astype("int")

			print("poss: ", ini_x, ini_y, end_x, end_y)
			label = CLASSES[id_label] + ": " + str(confidence*100)

			print("ith_label", CLASSES[id_label])
			object_detected.update({"object_" + str(i+1):CLASSES[id_label], "confidence_" + str(i+1):str(confidence)})

			# Print on the image
			cv2.rectangle(image, (ini_x, ini_y), (end_x, end_y), COLORS[id_label], 2)

			if ini_y - 10 > 10:
				y = ini_y - 10
			else:
				y = ini_y + 10

			cv2.putText(image, label, (ini_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[id_label], 2)
			succeeded = 1
		else:
			succeeded = 0



	# Constructing response
	result = {}

	if succeeded == 0:
		result.update({"Detected": False})
	else:
		image_dim = {
			"weight": w,
			"height": h,
		}
		#object_det = {'object': label, 'confidence':}
		result.update({"Detected": True, "Dimensions": image_dim, "Objects": object_detected})

	return result