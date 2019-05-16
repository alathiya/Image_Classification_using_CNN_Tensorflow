# Flower Image Classification using Tensorflow hub

In this notebook, CNN is created from scratch with FC layer for classification for 5 different species of flowers on flowers dataset.
Model is further optimized and overfitting is reduced through data augmentation and hyperparameter tunning.   

## Project Dependencies 

This project requires following libraries to be imported. 

	- glob
	- os
	- shutil
	- tensorflow
	- matplotlib
	- numpy
	


## Data
Flowers dataset is downloaded from URL "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz".
After downloading images for 5 categories of flowers, images are moved into 'train' and 'valid' folders such that 80% of images
go to training set and remaining 20% go to validation set. 

## DataAugmentation
ImageDataGenerator is use to create transformation that recales the images by 255, applies horizontal flip, rotation range, 
zoom range. This transformation is applied to all images in training dataset. Please note in validation dataset only rescaling
transformation is performed.  

## Implementation

CNN is create with 3 convolution layers followed by Max Pooling. Last Layer from CNN is flatten before feeding to FC layer. The CNN 
outputs class probabilities based on 5 classes which is done by the softmax activation function. Dropout probability of 20% is added
after last layer of CNN before flattening to reduce overfitting. 

Model is trained using fit_generator function instead of the usual fit function. Fit_generator function is used because we are using the 
ImageDataGenerator class to generate batches of training and validation data for our model. Model is trained for 80 epochs. 

## Project Observations:

Training and Validation Accuracy and Loss are plotted to analyse model complexity. Further experimentation is done with data augmentation 
and hyper parameter tunning to improve accurcy of model. After experimentation, final model accuracy of ~80% is acheieved. 

### Below are some of the further improvements I think can be done.

	- Adding more layers to fully connected classification layer to improve accuracy. 
	- Overfitting can be reduced by adding Dropout layers, L1/L2 Regularization. 
	- We can try changing different hyperparameter like Epochs, Batch Size, learning rate etc to further improve accuracy of model. 
