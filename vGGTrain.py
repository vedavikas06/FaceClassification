import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense,Dropout
import matplotlib.pyplot as plt
from keras.applications import VGG16

train_data_dir = 'data3D/training'

validation_data_dir = 'data3D/validation'

train_samples = 301

validation_samples = 151

epoch = 5

img_width, img_height = 100, 100

# ** Model Begins **
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(100, 100, 3))

conv_base.trainable = True

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    else:
        layer.trainable = False

conv_base.summary()

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))


model.summary()
# ** Model Ends **

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255)

validation_datagen = ImageDataGenerator(
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        #color_mode = 'grayscale',
        batch_size=10,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
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
        nb_val_samples=validation_samples,
        )

model.save_weights('neuralnetVGG.h5')
model.save('savedVGGModel.h5')

test_generator = test_datagen.flow_from_directory(
        'data3D/testing',
        target_size=(100, 100),
        batch_size=1,
        class_mode = None,  # only data, no labels
        shuffle = False)  # keep data in same order as labels

# test_loss,test_acc = model.evaluate_generator(test_generator,steps = 1)
# print('test accuracy :',test_acc)
# print('test loss :',test_loss)
probabilities = model.predict_generator(test_generator, 66)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
y_true = np.array([0] * 20 + [1] * 25 + [2] * 21)
#y_pred = probabilities > 0.5
print(probabilities)
y_pred = np.asarray(probabilities)
y_pred = np.argmax(probabilities,axis=1)

print(y_pred)

print(y_true)

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




