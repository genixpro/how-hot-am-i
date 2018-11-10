from flask import Flask, request
import keras
import keras.models
from scipy.misc import imread, imresize
from io import BytesIO
import base64
from flask_cors import CORS
import numpy
import json
import cv2
import math
import skimage.util
from .face_cropper import cropFaceAndPad

app = Flask(__name__)
CORS(app)

beautyModel = keras.models.load_model('beauty_model')
beautyModel._make_predict_function()


@app.route("/", methods=["POST"])
def analyzePhoto():
    imageBase64 = json.loads(request.data)['image']
    imageBase64 = imageBase64.split(',')[1]

    stream = BytesIO()
    stream.write(base64.b64decode(imageBase64))
    stream.seek(0)

    imageData = imread(stream)

    imageData = imageData[:, :, :3] # remove alpha channel if it exists

    imageData = numpy.array(imageData, dtype=numpy.float32)

    imageData = cropFaceAndPad(imageData)

    imageBatch = numpy.array([imageData])[:,:,:,:3]

    imageBatch = imageBatch * (1.0 / 255.0)

    score = min(1.0, float(beautyModel.predict(imageBatch)[0]) * 1.1)

    return json.dumps({"score": score})
