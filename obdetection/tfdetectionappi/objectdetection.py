from django.conf import settings

import os
import base64
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from tfdetectionappi import visualization


# Model preparation 
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

# Path to frozen detection graph. This is the model
PATH_TO_CKPT_ = MODEL_NAME + '/frozen_inference_graph.pb'

# List of strings that is used to add labels to boxes
PATH_TO_LABELS_ = os.path.join('data', 'mscoco_label_map.pbtxt')

# Num of objects that can recognize
NUM_CLASSES = 90

# Global path
PATH_TO_CKPT = os.path.join(settings.RESOURCE_ROOT, PATH_TO_CKPT_)
PATH_TO_LABELS = os.path.join(settings.RESOURCE_ROOT, PATH_TO_LABELS_)


# Load frozen tensorflow model
#def loading_model(model_path=PATH_TO_CKPT, label_path=PATH_TO_LABELS, num_class=NUM_CLASSES):
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

# Loading label maps
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#return detection_graph

# Detection
def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				'num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)
			if 'detection_masks' in tensor_dict:
				# Processing only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframing is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict,
									feed_dict={image_tensor: np.expand_dims(image, 0)})

			# Outputs are float32 np.arrays, so we have to convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]

	return output_dict

# Object detection wrapper
def detect_objects(image):
	# Image decoding from buffer
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	print("image type (image ready for TF) ", type(image), image_np.shape)

	#detection_graph = loading_model()
	# Actual detection
	output_dict = run_inference_for_single_image(image_np, detection_graph)
	#print("output_dict", output_dict)

	# Draw boxes on the image
	image_np, object_detected = visualization.visualize_boxes_and_labels_on_image_array(
		image_np,
		output_dict['detection_boxes'],
		output_dict['detection_classes'],
		output_dict['detection_scores'],
		category_index,
		instance_masks=output_dict.get('detection_masks'),
		use_normalized_coordinates=True,
		line_thickness=8)

	# Construction response
	result = {}

	print("object detected: ", object_detected)
	(h, w) = image_np.shape[:2]
	image_dim = {
			"weight": w,
			"height": h,
		}	
	result.update({"Detected": True, "Dimensions": image_dim, "Objects": object_detected})
	retval, buffer = cv2.imencode('.jpg', image_np)
	jpg_as_text = base64.b64encode(buffer)

	result.update({"Img64": jpg_as_text.decode("utf-8")})

	return result