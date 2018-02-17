from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import numpy as np
import os
import base64

from appi.detection import detection

# Endpoints definition
@csrf_exempt
def detect(request):
	if request.method == "GET":
		result = {"Object detection API": "Welcome to the API"}
		result.update({"API info": "Read carefully the following documentation"})
		result.update({"Description": "Send request via POST method to url"})
		result.update({"Url":"ec2-34-229-77-45.compute-1.amazonaws.com"})
		result.update({"POrt": 8000})
		result.update({"Endpoint": "/detect"})
		result.update({"Parameters": {"data": "Base 64 image encoded", \
			"type": "Type of data (ex. image)"}})
		result.update({"Response": {"Detected": "True or False", "Dimensions": \
			"Dimensions of image", "Objects": "List of recognized objects"},})

	if request.method == "POST":
		# Dealing with encoded data
		file_encoded = request.POST.get('data', None)
		file_decoded = base64.b64decode(file_encoded)
		#print("file_decoded", type(file_decoded))
		image = np.fromstring(file_decoded, dtype=np.uint8)
		print("decoded image", type(image))

		# Detect objects
		result = detection(image)

	# Return a Json response
	return JsonResponse(result)