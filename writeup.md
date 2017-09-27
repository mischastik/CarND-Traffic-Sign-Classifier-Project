#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/signs.jpg "Images from the training dataset"
[image2]: ./writeup/labels_histogram.png "Label distibutions."
[image3]: ./my-signs/12_1.png "Traffic Sign 1"
[image4]: ./my-signs/13_1.png "Traffic Sign 1"
[image5]: ./my-signs/14_1.png "Traffic Sign 2"
[image6]: ./my-signs/15_1.png "Traffic Sign 3"
[image7]: ./my-signs/35_1.png "Traffic Sign 4"
[image8]: ./my-signs/36_1.png "Traffic Sign 5"
[image9]: ./my-signs/37_1.png "Traffic Sign 6"

** Rubric Points
*** Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mischastik/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

Data exploration:

I used the some basic output to determine summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43


Here is a visualization of an excerpt of the data set images:

![Training set][image1]

The histogram shows that the distribution of labels in the training and validation set are very similar:

![Label distribution][image2]

## Design and Test a Model Architecture

Preprocessing:

I didn't convert the images to grayscale because color is a relevant property for traffic signs.
There were significant differences in image brightness, so I decided to normalize the images with respect to their maximum value.
I didn't generate additional data because the images already capture a lot of variablility in terms of image artifacts, backgrounds and environmental conditions.

Model Architecture:
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| Leaky RELU					|												|
| Dropout					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| Leaky RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| Leaky RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten				|												|
| Fully connected		| 800x240        									|
| Leaky RELU					|												|
| Dropout					|												|
| Fully connected		| 240x120        									|
| Leaky RELU					|												|
| Dropout					|												|
| Fully connected		| 120x43        									|
| Leaky RELU					|												|
| Dropout					|												|
| Softmax				|         									|
 
## Training
I chose a learning rate of 0.001, a keep probability of 85%. The variables were initialized with the Xavier initializer. Optimization was done with the Adam Optimizer on cross entropy.
A batch size of 512 and 10 epochs turned out to be an efficent choice. 

## Model Improvement
 I started with a very basic architecture close to the original LeNet. The first results were not satisfying and experimenting with hyperparamters didn't improve the results a lot.
Then I changed the image normalization to the apporach which uses the image maximum instead of 255 which imroved the results.
Next I added dropout layers, leaky RELUs and batch normalization which caused a decrease in performance.
I removed batch normalization which caused a huge improvement.
The training accuracy was still significantly higher than the validation accuracy so I decided to increase the model complexity by adding another convolution layer. Another optimization of the hyperparameters led to satisfying results.

My final results are:
Training accuracy: 0.997
Validation accuracy: 0.967
Test accuracy: 0.944

## Test a Model on New Images

The web search turned out to be tedious. After I found  only two proper images on the web, I took desperate measures: **I WENT OUTSIDE**. Outside I found lots of traffic signes which I recorded with my cell phone camera. 

![sign 1][image3] ![sign 2][image4] ![sign 3][image5] 
![sign 4][image6] ![sign 5][image7] ![sign 6][image8]

The images I found and the ones I recorded are all of high quality, so the detection results are good and the top softmax probability for each estimate was very high (>99% in all cases) and all estimates were correct.

