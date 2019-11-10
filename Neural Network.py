# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 08:27:19 2019

@author: Rajvindra
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.preprocessing import image

classifier=Sequential()
# Step 1-Convolution
classifier.add(Conv2D(64,kernel_size=3,strides=1,input_shape=(64,64,3),activation='relu'))
classifier.add(Conv2D(64,kernel_size=3,strides=2,activation='relu'))
classifier.add(Dropout(0.5))
#Step 4 - Full Connection
# Adding a second convolution layer
classifier.add(Conv2D(128,kernel_size=3,strides=1,activation='relu'))
classifier.add(Conv2D(128,kernel_size=3,strides=2,activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Conv2D(256,kernel_size=3,strides=1,activation='relu'))
classifier.add(Conv2D(256,kernel_size=3,strides=2,activation='relu'))
classifier.add(Dropout(0.5))

# Step-3 Flattening
classifier.add(Flatten())

#Step 4 Full connection
classifier.add(Dense(output_dim=64,activation='relu'))

classifier.add(Dense(output_dim=27,activation='softmax'))

#Compiling the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(validation_split=0.1)



training_set=train_datagen.flow_from_directory('data',
                                               target_size=(64,64),
                                               batch_size=64,
                                               class_mode='categorical',
                                               subset='training')
test_set=train_datagen.flow_from_directory('data',
                                         target_size=(64,64),
                                         batch_size=64,
                                         class_mode='categorical',
                                         subset='validation')
classifier.fit_generator(training_set,

                        nb_epoch=5,
                        validation_data=test_set
                          )                                          
classifier.save('model5.h5')
