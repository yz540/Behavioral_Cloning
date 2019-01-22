import csv
import cv2
import numpy as np
from scipy import ndimage
# Setup Keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def processImg(img_path):
    image = ndimage.imread(img_path)
    cropped_img = image[65:image.shape[0]-25, :, :]
    return cropped_img

def generator(path, samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # read in images from center, left and right cameras
                image_center = processImg(path + batch_sample[0].lstrip())
                image_left = processImg(path + batch_sample[1].lstrip())
                image_right = processImg(path + batch_sample[2].lstrip())
                images.extend([image_center, image_left, image_right])
                
                # steering angle for the center image_center
                # correct the steering angle for left and right camera images as if they were from the center camera
                steering_angle_center = float(batch_sample[3])
                correction = 0.2
                steering_left = steering_angle_center + correction
                steering_right = steering_angle_center - correction
                angles.extend([steering_angle_center, steering_left, steering_right])

            images_flipped = [np.fliplr(image) for image in images]
            angles_flipped = [-angle for angle in angles]
            
            # add flipped images
            images.extend(images_flipped)
            angles.extend(angles_flipped)
        
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
def load_data(path):
    samples = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header of sample data
        for line in reader:
            samples.append(line)
    return samples


def nn_model(train_generator, validation_generator):
    # compile and train the model using the generator function
    model = Sequential()
    
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(70,320,3)))
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
#     model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
    return model

sample_data_path = "../../../opt/carnd_p3/data/"
samples = load_data(sample_data_path)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(sample_data_path, train_samples, batch_size=32)
validation_generator = generator(sample_data_path, validation_samples, batch_size=32)
model = nn_model(train_generator, validation_generator)
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'