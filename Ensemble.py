# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:43:31 2020

@author: Aaryan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

nets = 15
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()
    
    # Step 1 - Convolution
    model[j].add(Conv2D(64, (5, 5), input_shape = (64, 64, 1), activation = 'relu'))
    model[j].add(Conv2D(64, (5, 5), activation='relu'))
    model[j].add(MaxPooling2D(pool_size=(2, 2)))
    model[j].add(Dropout(rate = 0.25))
     
    # Adding a third convolutional layer
    model[j].add(Conv2D(64, (3, 3), activation='relu'))
    model[j].add(MaxPooling2D(pool_size=(2, 2)))
    model[j].add(Dropout(rate = 0.25))
    
    # Step 3 - Flattening
    model[j].add(Flatten())
    
    # Step 4 - Full connection
    model[j].add(Dense(units = 256, activation = 'relu'))
    model[j].add(Dropout(rate = 0.5))
    
    model[j].add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    model[j].compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
epochs = 3

total_image_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.1)


for j in range(nets):
    training_set = total_image_datagen.flow_from_directory('TotalDataset',
                                                     target_size = (64, 64),
                                                     batch_size = 16,
                                                     subset = "training",
                                                     class_mode = 'binary',
                                                     color_mode = 'grayscale')
    
    test_set = total_image_datagen.flow_from_directory('TotalDataset',
                                                target_size = (64, 64),
                                                batch_size = 16,
                                                subset = "validation",
                                                class_mode = 'binary',
                                                color_mode = 'grayscale')

    history[j] = model[j].fit_generator(training_set,
        epochs = epochs, steps_per_epoch = 673//16,  
        validation_data = test_set, callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
    
    plt.plot(history[j].history['val_accuracy'], label = ("val_acc " + str(j)))
    
plt.title('Progres Over Epochs')
plt.ylabel('Loss/Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc='best')
plt.show()


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set1 = test_datagen.flow_from_directory('TestSet',
                                            target_size = (64, 64),
                                            batch_size = 16,
                                            class_mode = 'binary',
                                            color_mode = 'grayscale',
                                            shuffle = False)
                                             
model[5].evaluate_generator(test_set1)  

#Saving Models
from keras.models import load_model

model[5].save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

model89 = load_model('my_model.h5')
model89.evaluate_generator(test_set1)
