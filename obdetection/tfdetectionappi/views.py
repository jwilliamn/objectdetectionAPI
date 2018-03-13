from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import numpy as np
import base64

from tfdetectionappi.objectdetection import detect_objects

# Endpoints.
@csrf_exempt
def detect(request):
	if request.method == "POST":
		data_encoded = request.POST.get('data', None)
		data_decoded = base64.b64decode(data_encoded)

		image = np.fromstring(data_decoded, dtype=np.uint8)
		print("Type of image (after decoding): ", type(image))

		result = detect_objects(image)

	return JsonResponse(result)