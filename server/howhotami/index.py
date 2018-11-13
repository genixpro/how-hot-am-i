from flask import Flask, request, Response
import keras
import keras.models
from scipy.misc import imread, imresize, imsave
from azure.storage.blob import BlockBlobService, PublicAccess
from io import BytesIO
import base64
from flask_cors import CORS
import numpy
import os.path
import json
import datetime
import random
import math
import skimage.util
from .face_cropper import cropFaceAndPad

app = Flask(__name__)
CORS(app)

beautyModel = keras.models.load_model('beauty_model')
beautyModel._make_predict_function()

# Create the BlockBlockService that is used to call the Blob service for the storage account
azureBlobStorage = BlockBlobService(account_name='howhotamiphotos',
                                    account_key='taRUidmAdF/rJnkaCLlOuYDUVowZG8gT/iOnUqUVgnRMzw6ynlFvyhvMfwVi+7mIT/4ES6NuT3oMWFZELgqkvA==')

fileCache = {}

@app.route("/process", methods=["POST"])
def analyzePhoto():
    imageBase64 = json.loads(request.data)['image']
    imageBase64 = imageBase64.split(',')[1]

    stream = BytesIO()
    stream.write(base64.b64decode(imageBase64))
    stream.seek(0)

    name = datetime.datetime.now().isoformat() + "-" + str(random.randint(0, 100)) + ".jpg"
    azureBlobStorage.create_blob_from_bytes('photos', name, stream.read())

    stream.seek(0)
    imageData = imread(stream)

    imageData = imageData[:, :, :3] # remove alpha channel if it exists

    imageData = numpy.array(imageData, dtype=numpy.float32)

    imageData = cropFaceAndPad(imageData)

    imageBatch = numpy.array([imageData])[:,:,:,:3]

    imageBatch = imageBatch * (1.0 / 255.0)

    score = max(0.0, min(0.999, float(beautyModel.predict(imageBatch)[0]) * 1.05 + 0.10))

    outImage = BytesIO()
    imsave(outImage, imageBatch[0] * 255.0, 'jpeg')

    encodedimage = str(base64.b64encode(outImage.getbuffer()), 'ascii')

    return json.dumps({"score": score, "image": encodedimage})


def getStaticFileData(subpath):
    if subpath in fileCache:
        return fileCache[subpath]

    filePath = os.path.join('/home/bradley/how-hot-am-i/client/build', subpath)
    print(filePath)
    if os.path.exists(filePath):
        with open(filePath, 'rb') as staticFile:
            data = staticFile.read()
            fileCache[subpath] = data
            return data
    else:
        return "Not Found"


@app.route('/')
def showIndex():
    return getStaticFileData('index.html')

@app.route('/<path:subpath>')
def show_subpath(subpath):
    subpath = subpath.replace("..", "")

    if 'png' in subpath:
        return Response(getStaticFileData(subpath), mimetype='image/png')
    elif 'svg' in subpath:
        return Response(getStaticFileData(subpath), mimetype='image/svg+xml')
    elif 'js' in subpath:
        return Response(getStaticFileData(subpath), mimetype='application/javascript')
    else:
        return getStaticFileData(subpath)

@app.route('/static/js/<path:subpath>')
def show_static_js_subpath(subpath):
    subpath = subpath.replace("..", "")
    return Response(getStaticFileData(os.path.join('static/js', subpath)), mimetype='application/javascript')

@app.route('/static/css/<path:subpath>')
def show_static_css_subpath(subpath):
    subpath = subpath.replace("..", "")
    return Response(getStaticFileData(os.path.join('static/css', subpath)), mimetype='text/css')

@app.route('/static/media/<path:subpath>')
def show_static_media_subpath(subpath):
    subpath = subpath.replace("..", "")
    return Response(getStaticFileData(os.path.join('static/media', subpath)), mimetype='image/svg+xml')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
