import csv
import cv2
import numpy as np
from scipy import ndimage

def load_data(path):
    lines = []
    # read image path from the csv file
    with open(sample_data_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header of sample data
        for line in reader:
            lines.append(line)
#             print(line)

    # read images
    images = []
    steering_angles = []
    for line in lines:
        image = ndimage.imread(path + line[0])
        images.append(image)
        steering_angle = float(line[3])
        steering_angles.append(steering_angle)
        # add flipped images
        image_flipped = np.fliplr(image)
        images.append(image)
        steering_angle_flipped = -steering_angle
        steering_angles.append(steering_angle_flipped)
        
    X_train = np.array(images)
    y_train = np.array(steering_angles)
    return X_train, y_train

# Setup Keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def nn_model(X_train, y_train):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    return model

sample_data_path = "../../../opt/carnd_p3/data/"
# sample_data_path = "C:/Workspace/CamU/Projects/bhvCloning/data"
X_train, y_train = load_data(sample_data_path)
model = nn_model(X_train, y_train)
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'