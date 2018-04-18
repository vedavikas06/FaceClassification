import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
import matplotlib.pyplot as plt
import sys
from keras.applications import VGG16

img_width, img_height = 100, 100

def create_model():


 # ** Model Begins **
 conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(100, 100, 3))

 conv_base.trainable = True

 for layer in conv_base.layers:

    if layer.name == 'block5_conv1':
        layer.trainable = True
    if layer.name == 'block4_conv1':
        layer.trainable = True
    else:
        layer.trainable = False

 conv_base.summary()

 model = Sequential()
 model.add(conv_base)
 model.add(Flatten())
 model.add(Dense(500, activation='relu'))
 model.add(Dropout(0.5))

 model.add(Dense(6, activation='softmax'))


 model.summary()

 return model

#change the image name
img = cv2.imread('down.jpg')

#predicting the image


#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (img_width, img_height))

model = create_model()

model.load_weights('./neuralnetVGG.h5')

arr = numpy.array(img).reshape((img_width,img_height,3))

arr = numpy.expand_dims(arr, axis=0)

prediction = model.predict(arr)[0]

print(prediction)

bestclass = ''

bestconf = -1

best = ['korean','indian','usa','indonesia','philippines','kenya']
for n in [0,1,2,3,4,5]:
	if (prediction[n] > bestconf):
		bestclass = n
		bestconf = prediction[n]

print ('This image is a ' + best[bestclass])
