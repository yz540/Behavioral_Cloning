**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model I used is a modified LeNet model. Compared to the LeNet model, I added the lambda_1 layer where data is normalized and two dropout layers to reduce overfitting, removed one fully-connected layer to decrease training time. The corresponding code is in the method nn_model() in model.py (lines 70-87).

The summary of the model is shown as follows.

|Layer (type)        |         Output Shape       |       Param #   |
|:---------------------:|:---------------------------------------------:| 
|lambda_1 (Lambda)    |        (None, 70, 320, 3)   |     0       |
|conv2d_1 (Conv2D)    |       (None, 66, 316, 6)    |    456      |
|max_pooling2d_1 (MaxPooling2) | (None, 33, 158, 6) |       0     |    
|conv2d_2 (Conv2D)    |        (None, 29, 154, 6)    |    906     | 
|max_pooling2d_2 (MaxPooling2) | (None, 14, 77, 6)  |       0     |  
|flatten_1 (Flatten)  |        (None, 6468)          |    0       |  
|dropout_1 (Dropout)  |        (None, 6468)          |    0       |  
|dense_1 (Dense)      |        (None, 84)            |    543396  |  
|dropout_2 (Dropout)  |        (None, 84)            |    0       |  
|dense_2 (Dense)      |        (None, 1)             |    85      |  

Total params: 544,843  
Trainable params: 544,843  
Non-trainable params: 0  

#### 4. Appropriate training data
I am not a good driver and often drive the car to water or grass, so I used the sample data provided by Udacity. To increase the training data, I used the images from centre, left and right cameras and flipped them all, which gives 6 times the original data.
The corresponding code is in process_data() method.

### Model Architecture and Training Strategy
The overall strategy for deriving a model architecture was to:
* start with a well-know model on sample data 
* and then use transfer learning to fine tune the model on a new set of data several times 
* and then obtain the final model.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it has shown its capacity in the demonstration in the Udacity course.

The first data set I used is the images from the centre camera of the sample data. I split my image and steering angle data into a training (6428 images, 80%) and validation set (20%). I found that the model had a low loss on the training set but a high loss on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I used images from left and right cameras and flipped all the images to improve the genericity. There are 8036 * 6 images to train, which consumes too much memory if load at once. I used a generator to read 32 images at a time, producing 192 data for training each time (see code of the generator in lines 31-68) on the fly. The training was finished within 1h30 on a remote GPU and a model.h5 was saved. But when using the saved model to drive the car in drive.py, it was stuck at the entrance of the bridge because it was not at the middle of the road. 

Note that the training images were cropped first as the input of the model. Therefore, in the drive.py the prediction of steering angle is also on a cropped image as shown in line 64 in drive.py.

To handle the turns, I collected additional around 1600 data by controlling the car in the simulator with sharp turns struggling to stay on the road. And used transfer learning to fine tune the model, see the commented code between lines (89-96, 100-101, and 111-114), on the new data set. I used fine tuning because similar data and big data set. The time of training is relatively short than the previous training. In the simulator, I could see that the result model managed to finish almost one full track.

The new data set is not good enough because I almost drive the car off the road using keyboard. So I decided to abandon my own data. Then I modified the model by adding two dropout layers to reduce overfitting. It worked and the overfitting is reasonably low, see the training loss and validation loss are very close. The accuracy didn't improve much from Epoch 2 to Epoch 3, so 3 epochs is the good enough choice for the training.

Epoch 1/3  
6428/6428 [==============================] - 1308s - loss: 0.0176 - val_loss: 0.0160  
Epoch 2/3  
6428/6428 [==============================] - 1287s - loss: 0.0146 - val_loss: 0.0156  
Epoch 3/3  
6428/6428 [==============================] - 1285s - loss: 0.0142 - val_loss: 0.0153  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The Adam optimizer was used in the training, so no need to specify learning rate.