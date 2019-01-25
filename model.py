import csv
import cv2
import numpy as np
from scipy import ndimage
# Setup Keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model
from keras import optimizers

def load_data(path):
    samples = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header of sample data, comment this line if not sample data
        for line in reader:
            samples.append(line)
    return samples

def processImg(img_path):
    image = ndimage.imread(img_path)
    # crop the image to keep only the road part
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
                # and crop them to keep only the road part
                image_center = processImg(path + './IMG/'+ batch_sample[0].split('/')[-1])
                image_left = processImg(path + './IMG/' + batch_sample[1].split('/')[-1])
                image_right = processImg(path + './IMG/' + batch_sample[2].split('/')[-1])
                images.extend([image_center, image_left, image_right])
                
                # steering angle for the center image_center
                steering_angle_center = float(batch_sample[3])
                # correct the steering angle for left and right camera images as if they were from the center camera
                correction = 0.2
                steering_left = steering_angle_center + correction
                steering_right = steering_angle_center - correction
                angles.extend([steering_angle_center, steering_left, steering_right])

            # data augmentation by flipping images and reverse steering angles
            images_flipped = [np.fliplr(image) for image in images]
            angles_flipped = [-angle for angle in angles]
            
            # add flipped images and steering angles
            images.extend(images_flipped)
            angles.extend(angles_flipped)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # produce 6*batch_size images to train each time
            yield shuffle(X_train, y_train)

def nn_model(train_generator, validation_generator):
    # add normalization layer and dropout layers to the LeNet model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(70,320,3)))
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(84))
    model.add(Dropout(.5))
    model.add(Dense(1))
    print(model.summary())
    # compile and train the model using the generator function
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 1)
    return model

def transfer_learning(model_path, train_generator, validation_generator):
    # Load the previously trained model and initialize the weights from the pretrained model
    ft_model = load_model(model_path)
    ft_model.load_weights(model_path)
    # compile and train the model on the new data set
    ft_model.compile(loss='mse', optimizer='adam')
    ft_model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)
    return ft_model

# first round trian
sample_data_path = "../../../opt/carnd_p3/data/"
# # fine tune data
# sample_data_path = "../../../opt/carnd_p3/train_data/"
samples = load_data(sample_data_path)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(sample_data_path, train_samples, batch_size=32)
validation_generator = generator(sample_data_path, validation_samples, batch_size=32)

# Train the model from scratch
model = nn_model(train_generator, validation_generator)
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'

# # Use transfer learning with a pretrained model
# model_path = 'model.h5'
# ft_model = transfer_learning(model_path, train_generator, validation_generator)
# ft_model.save('ft_model.h5')