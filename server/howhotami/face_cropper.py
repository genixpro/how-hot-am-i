from flask import Flask, request
import keras
import keras.models
from scipy.misc import imread, imresize
from io import BytesIO
import base64
from flask_cors import CORS
import numpy
import json
import random
import cv2
import math
import tensorflow as tf
import skimage.util

from .tensorflow_face_detection.inference_usbCam_face import TensoflowFaceDector

app = Flask(__name__)
CORS(app)

with tf.device('/cpu:0'):
    beautyModel = keras.models.load_model('beauty_model')
    beautyModel._make_predict_function()

    PATH_TO_CKPT = './howhotami/tensorflow_face_detection/model/frozen_inference_graph_face.pb'
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

def cropFace(imageData):
    (boxes, scores, classes, num_detections) = tDetector.run(imageData)

    box = boxes[0][0]
    xmin, ymin, xmax, ymax = tuple(box)

    xmin = int(xmin * imageData.shape[1])
    xmax = int(xmax * imageData.shape[1])

    ymin = int(ymin * imageData.shape[0])
    ymax = int(ymax * imageData.shape[0])

    circleCenterX = int(xmax/2 + xmin/2)
    circleCenterY = int(ymax/2 + ymin/2)

    circleRadius = int(max(ymax - ymin, xmax - xmin) * 0.5 * 1.2)

    mask = numpy.zeros((imageData.shape[0], imageData.shape[1], 3),dtype=numpy.uint8)

    cv2.circle(mask, (circleCenterX, circleCenterY), circleRadius, (255, 255, 255), -1, 8, 0)

    #cv2.imwrite(argv[2],mask)
    out = imageData * mask
    white = 255 * (1.0 - mask)

    imageData = out + white

    xmin = circleCenterX - circleRadius
    xmax = circleCenterX + circleRadius

    ymin = circleCenterY - circleRadius
    ymax = circleCenterY + circleRadius

    xmin = max(0, int(xmin))
    xmax = min(imageData.shape[0], int(xmax))

    ymin = max(0, int(ymin))
    ymax = min(imageData.shape[1], int(ymax))

    return imageData[ymin:ymax, xmin:xmax, :]


def cropFaceAndPad(imageData):
    outWidth = 350
    outHeight = 350

    imageData = cropFace(imageData)

    squareSize = max(imageData.shape[0], imageData.shape[1])

    padHeight = (max(0, int(math.floor((squareSize-imageData.shape[0])/2))), max(0, int(math.ceil((squareSize-imageData.shape[0])/2))))
    padWidth = (max(0, int(math.floor((squareSize-imageData.shape[1])/2))), max(0, int(math.ceil((squareSize-imageData.shape[1])/2))))

    imageData = skimage.util.pad(imageData, pad_width=[padHeight, padWidth, (0, 0)], mode='constant', constant_values=255)

    imageData = imresize(imageData, (outWidth, outHeight))

    # Convert to greyscale
    imageData = numpy.repeat(numpy.reshape(numpy.average(imageData, axis=2), newshape=(outWidth, outHeight, 1)), 3, axis=2)

    return numpy.array(imageData, dtype=numpy.float64)

