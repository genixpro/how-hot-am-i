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

from .tensorflow_face_detection.inference_usbCam_face import TensoflowFaceDector

app = Flask(__name__)
CORS(app)

beautyModel = keras.models.load_model('beauty_model')
beautyModel._make_predict_function()

PATH_TO_CKPT = './howhotami/tensorflow_face_detection/model/frozen_inference_graph_face.pb'
tDetector = TensoflowFaceDector(PATH_TO_CKPT)


@app.route("/", methods=["POST"])
def analyzePhoto():
    imageBase64 = json.loads(request.data)['image']
    imageBase64 = imageBase64.split(',')[1]

    stream = BytesIO()
    stream.write(base64.b64decode(imageBase64))
    stream.seek(0)

    imageData = imread(stream)

    cv2.imshow("face0", imageData)
    cv2.waitKey(100)

    (boxes, scores, classes, num_detections) = tDetector.run(imageData)

    box = boxes[0][0]
    xmin, ymin, xmax, ymax = tuple(box)

    xmin -= 0.10
    xmax += 0.10
    ymin -= 0.15
    ymax += 0.15

    xmin = max(0, int(xmin * imageData.shape[0]))
    xmax = min(imageData.shape[0], int(xmax * imageData.shape[0]))

    ymin = max(0, int(ymin * imageData.shape[1]))
    ymax = min(imageData.shape[1], int(ymax * imageData.shape[1]))

    imageData = imageData[xmin:xmax, ymin:ymax]

    # print(imageData.shape)
    cv2.imshow("face1", imageData)
    cv2.waitKey(100)

    squareSize = max(imageData.shape[0], imageData.shape[1])

    padHeight = (max(0, int(math.floor((squareSize-imageData.shape[0])/2))), max(0, int(math.ceil((squareSize-imageData.shape[0])/2))))
    padWith = (max(0, int(math.floor((squareSize-imageData.shape[1])/2))), max(0, int(math.ceil((squareSize-imageData.shape[1])/2))))
    imageData = skimage.util.pad(imageData, pad_width=[padHeight, padWith, (0, 0)], mode='constant', constant_values=255)

    cv2.imshow("face2", imageData)
    cv2.waitKey(100)

    imageData = imresize(imageData, (224, 224))

    # imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)

    # print(imageData.shape)
    cv2.imshow("face3", imageData)
    cv2.waitKey(100)

    imageBatch = numpy.array([imageData])[:,:,:,:3]

    imageBatch = imageBatch * (1.0 / 255.0)

    score = float(beautyModel.predict(imageBatch)[0])

    return json.dumps({"score": score})
