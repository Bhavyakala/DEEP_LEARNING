# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# importing the libraries
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense 

# initialising the CNN
classifier = Sequential()

#step - 1 convolution
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))

#step - 2 pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step - 3 flattening
classifier.add(Flatten())

#step - 4 fullconnection (ANN)
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the cnn to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) # here image transformations are applied

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary') # target_size = size of images we feed in CNN, batch_size same as ANN

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=8000, #no. of samples in dataset
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

    





















