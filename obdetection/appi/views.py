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