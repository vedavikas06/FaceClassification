import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
import matplotlib.pyplot as plt
import sys

img_width, img_height = 100, 100

#model can we written here for testing
def create_model():
 
 model = Sequential()
 model.add(Convolution2D(20, 3, 3, activation='relu', input_shape=(img_width, img_height,3)))
 model.add(MaxPooling2D(2, 2))

 model.add(Convolution2D(30, 5, 5, activation='relu'))
 model.add(MaxPooling2D(2, 2))

 model.add(Flatten())
 model.add(Dense(1000, activation='relu'))
 model.add(Dropout(0.5))

 model.add(Dense(6, activation='softmax'))
 model.summary()
 return model


img = cv2.imread('down.jpeg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (img_width, img_height))
model = create_model()
model.load_weights('./neuralnet3.h5')
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
