import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
import matplotlib.pyplot as plt

train_data_dir = 'data3D/training'

validation_data_dir = 'data3D/validation'

train_samples = 301

validation_samples = 151

epoch = 10

img_width, img_height = 100, 100


# ** Model Begins **
model = Sequential()
model.add(Convolution2D(20, 3, 3, activation='relu', input_shape=(img_width, img_height,3)))
model.add(MaxPooling2D(2, 2))

model.add(Convolution2D(30, 5, 5, activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))
model.summary()
# ** Model Ends **

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        #color_mode = 'grayscale',
        batch_size=10,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        #color_mode = 'grayscale',
        batch_size=10,
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        samples_per_epoch=train_samples,
        nb_epoch=epoch,
        validation_data=validation_generator,
        nb_val_samples=validation_samples)

model.save_weights('neuralnet3.h5')
generator = test_datagen.flow_from_directory(
        'data3D/testing',
        target_size=(100, 100),
        batch_size=1,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

probabilities = model.predict_generator(generator, 66)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
y_true = np.array([0] * 20 + [1] * 25 + [2] * 21)
#y_pred = probabilities > 0.5
print(probabilities)
y_pred = np.argmax(probabilities,axis=1)

#print(np.shape(probabilities))
print(confusion_matrix(y_true, y_pred))

print(accuracy_score(y_true, y_pred))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




