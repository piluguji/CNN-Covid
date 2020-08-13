# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

"""batches per weights updated"""
batch_sizeTrain = 16
'''what are the choices of possible images'''  
num_classes = 2   
'''how many times should it repeat'''
epochs = 100    

'''dimensions of the images'''
img_rows, img_cols = 64, 64

"""Size"""
TrainSize = 606
TestSize =  140


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (img_rows, img_cols, 1), activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(rate = 0.3))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('TrainingSet',
                                                 target_size = (img_rows, img_cols),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 color_mode = 'grayscale')

test_set = test_datagen.flow_from_directory('TestSet',
                                            target_size = (img_rows, img_cols),
                                            batch_size = 32,
                                            class_mode = 'binary',
                                            color_mode = 'grayscale')
classifier.fit_generator(training_set,
                             steps_per_epoch = TrainSize/batch_sizeTrain,  
                             epochs = epochs,
                             validation_data = test_set,
                             validation_steps = 1)


"""def runClassifier(model, batchSize, epoch):
    steps_per_epoch = TrainSize/batchSize
    model.fit_generator(training_set,
                             steps_per_epoch = TrainSize/batchSize,  
                             epochs = epoch,
                             validation_data = test_set,
                             validation_steps = TestSize/7)
    f = open("Log.txt", "a")
    log = "Batch Size: "  + str(batchSize) + ", Epoch: " + str(epoch)
    f.write(log)
    f.write("\n")
    print("-------------------------------------------------------------------------------------------------")"""