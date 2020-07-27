# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
file_train = r'C:\Users\mycp2cqj\Documents\CATD_PROJECT\Convolutional_Neural_Networks\dataset\trn_set'
file_test = r'C:\Users\mycp2cqj\Documents\CATD_PROJECT\Convolutional_Neural_Networks\dataset\tst_set'
training_set = train_datagen.flow_from_directory(file_train,
                                                 target_size = (64, 64),
                                                 batch_size = 7,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(file_test,
                                            target_size = (64, 64),
                                                batch_size = 7,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 50,
                         epochs =25,
                         validation_data = test_set,
                         validation_steps = 5)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\mycp2cqj\Documents\CATD_PROJECT\Convolutional_Neural_Networks\dataset\single_prediction\2.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#df = pd.DataFrame(result,columns = ['0','1','2','3','4','5','6','7','8','9'])
df = pd.DataFrame(result,columns = ['0','1','2'])

df
result2="none"

df2=np.where(df['age']>=50, 'yes', 'no')

result2  