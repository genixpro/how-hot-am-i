from keras.applications import MobileNetV2, InceptionV3
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import csv
from scipy.misc import imsave, imread
import os
import sklearn.metrics
import numpy
import os.path
import random
import json
from pprint import pprint
from .face_cropper import cropFaceAndPad

def trainModel():
    with open('howhotami/data/ratings.csv', 'rt') as file:
        images = []
        ratings = {}

        reader = csv.DictReader(file)

        for rating in reader:
            if rating['Filename'] not in ratings:
                ratings[rating['Filename']] = []
                images.append(rating['Filename'])

            score = float(rating['Rating'])
            # normalized = (score - 1)/4
            normalized = score/5

            ratings[rating['Filename']].append(normalized)

    for f in images:
        ratings[f] = numpy.average(ratings[f])

    sortedScores = sorted([ratings[f] for f in images])
    for f in images:
        ratings[f] = float(sortedScores.index(ratings[f])) / float(len(images))


    dataframe = pd.DataFrame(data={
                                'image': images,
                                'rating': [ratings[image] for image in images]
                             }, columns=['image', 'rating'])


    print(json.dumps(sorted([round(ratings[image]*1000)/1000 for image in images])))


    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.3,
        horizontal_flip=True,
        fill_mode='constant',
        cval=255,
        validation_split=0.2,
        preprocessing_function=cropFaceAndPad
    )

    batchSize = 32

    train_generator = datagen.flow_from_dataframe(dataframe=dataframe,
                                                  directory="howhotami/data/images",
                                                  # save_to_dir='training',
                                                  x_col="image",
                                                  y_col="rating",
                                                  has_ext=True,
                                                  class_mode="other",
                                                  target_size=(350, 350),
                                                  batch_size=batchSize,
                                                  subset='training')


    test_generator = datagen.flow_from_dataframe(dataframe=dataframe,
                                                  directory="howhotami/data/images",
                                                  # save_to_dir='testing',
                                                  x_col="image",
                                                  y_col="rating",
                                                  has_ext=True,
                                                  class_mode="other",
                                                  target_size=(350, 350),
                                                  batch_size=batchSize,
                                                  subset='validation')



    resnet = InceptionV3(include_top=False, pooling=None, input_shape=(350, 350, 3), weights='imagenet')
    model = Sequential()
    model.add(resnet)
    # model.add(Reshape(target_shape=(32, 224, 224, -1)))
    model.add(Conv2D(64, 3, activation='elu', kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02)))
    model.add(Reshape([-1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.layers[0].trainable = False

    print(model.summary())

    def mean_absolute_error(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true), axis=-1)

    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=5e-4), metrics=[mean_squared_error, mean_absolute_error])

    # print(train_generator.next())
    # print(test_generator.next())

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(len(images) * 0.8 / batchSize),
        epochs=5,
        validation_data=test_generator,
        validation_steps=int(len(images) * 0.2 / batchSize),
        workers=8,
        max_queue_size=50
    )


    model.layers[0].trainable = True
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics=[mean_squared_error, mean_absolute_error])
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(len(images) * 0.8 / batchSize),
        epochs=20,
        validation_data=test_generator,
        validation_steps=int(len(images) * 0.2 / batchSize),
        workers=8,
        max_queue_size=50
    )

    model.save('beauty_model')

    imagePredictedScores = []
    imageActualScores = []
    batch = []
    def finishBatch():
        nonlocal batch
        scores = model.predict(numpy.array([b[0] for b in batch]))

        imagePredictedScores.extend(scores)
        imageActualScores.extend([numpy.average(ratings[b[1]]) for b in batch])
        batch = []


    for image in images:
        imageData = imread(os.path.join('howhotami/data/images', image))

        imageData = imageData[:, :, :3]  # remove alpha channel if it exists

        imageData = numpy.array(imageData, dtype=numpy.float32)
        imageData = cropFaceAndPad(imageData)
        imageData = imageData * (1.0 / 255.0)

        batch.append((imageData, image))

        if len(batch) > 32:
            finishBatch()

    finishBatch()


    pprint(list(zip(imagePredictedScores, imageActualScores)))
