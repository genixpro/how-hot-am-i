from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import csv
from scipy.misc import imsave, imread
import os
import numpy
import os.path
import random
import json

with open('howhotami/data/ratings.csv', 'rt') as file:
    images = []
    ratings = {}

    reader = csv.DictReader(file)

    for rating in reader:
        if rating['Filename'] not in ratings:
            ratings[rating['Filename']] = []
            images.append(rating['Filename'])

        score = float(rating['Rating'])
        normalized = (score - 1)/4

        ratings[rating['Filename']].append(normalized)

dataframe = pd.DataFrame(data={
                            'image': images,
                            'rating': [float(numpy.average(ratings[image])) for image in images]
                         }, columns=['image', 'rating'])


print(json.dumps(sorted([round(float(numpy.average(ratings[image]))*1000)/1000 for image in images])))


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='constant',
    cval=255,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory="howhotami/data/images",
                                              # save_to_dir='training',
                                              x_col="image",
                                              y_col="rating",
                                              has_ext=True,
                                              class_mode="other",
                                              target_size=(224, 224),
                                              batch_size=32,
                                              subset='training')


test_generator = datagen.flow_from_dataframe(dataframe=dataframe,
                                              directory="howhotami/data/images",
                                              # save_to_dir='testing',
                                              x_col="image",
                                              y_col="rating",
                                              has_ext=True,
                                              class_mode="other",
                                              target_size=(224, 224),
                                              batch_size=32,
                                              subset='validation')



resnet = MobileNetV2(include_top=False, pooling="avg", input_shape=(224, 224, 3), weights='imagenet')
model = Sequential()
model.add(resnet)
# model.add(Reshape(target_shape=(32, 224, 224, -1)))
# model.add(Conv2D(32, 3, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu'))

model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.layers[0].trainable = False

print(model.summary())

model.compile(loss='mean_squared_error', optimizer=Adam())

# print(train_generator.next())
# print(test_generator.next())


batchSize = 32

model.fit_generator(generator=train_generator, steps_per_epoch=int(len(images) * 0.4 / batchSize), epochs=15, validation_data=test_generator, validation_steps=int(len(images) * 0.05 / batchSize))

model.save('beauty_model')