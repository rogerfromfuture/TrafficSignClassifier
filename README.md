#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. 
I shown one sign image and also the distributions of the unique sign number with the count for each number

![alt text][image1]

###Design and Test a Model Architecture

####1. Processing

I shuffle the data

####2. Describe how, and identify where in your code, you set up training, validation and testing data.

To cross validate my model, I load the training data into a training set and validation set respectively from different files in 2nd cell

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

The code for my final model is located in the 6th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6  	| 1x1 stride, valid  padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				    |
| Convolution  5x5x3x6 	| 5x5x16, valid  padding, outputs 10x10x16      |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				    |
| Flatten		        | outputs 400        							|
| Fully connected		| outputs 120        						    |
| RELU					|												|
| Fully connected		| outputs 84        						    |
| RELU					|												|
| Fully connected		| outputs 60        						    |
| RELU					|												|
| Fully connected		| outputs 43        						    |
| Softmax				| outputs 43       								|
 

####4. Describe how, and identify where in your code, you trained your model. 
The code for training the model is located in the 31st cell of the ipython notebook. 

To train the model, I used an learning rate of 0.001
Epoch 15
Batch size 64
Use cross entropy with adam opitimizer

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
The validation accuracy is around 89.9%
and test accuracy is around 89.6%

If an iterative approach was chosen:
I chose LENET as a base
But I found the epoch initial value is too small and batch size is too big so I tuned a bit. increased the accuracy a bit
And I also added one fully connected layer which helps to increase the accuracy a bit

The reason I choose Lenet because the classification of the traffic sign problem is very similar to hand written images
just the number of classifications varies.

###Test a Model on New Images
Added the 5 images and use the saved model to predict
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


The code for making predictions on my final model is located in the 77th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians     		| Pedestrians  									| 
| Road work     		| Road work 									|
| Speed limit (120km/h)	| Speed limit (120km/h)							|
| No entry	      		| No entry					 				    |
| Speed limit (80km/h)	| Speed limit (80km/h)   						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that Pedestrians (probability of 0.99 ), 
and the image does contain a Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99     		        | Pedestrians  									| 
| 5.65343164e-03    	| Dangerous curve to the right					|
| 3.60485306e-03        | General caution							    |
| 1.67435239e-13 	    | Road work				 				        |
| 1.38697399e-15        | Pedestrians   						        |

all images probabilities and predictions are
Top 5 probabilities are 
[[  9.90741730e-01   5.65343164e-03   3.60485306e-03   1.67435239e-13
    1.38697399e-15]
 [  1.00000000e+00   1.67094644e-10   1.56852989e-10   1.25374060e-11
    3.26056846e-12]
 [  9.92322266e-01   3.83022591e-03   3.35003785e-03   3.47833440e-04
    3.30624061e-05]
 [  1.00000000e+00   3.09686961e-43   0.00000000e+00   0.00000000e+00
    0.00000000e+00]
 [  1.00000000e+00   5.29178976e-08   8.47778653e-11   1.82002596e-14
    1.90034679e-15]]
    
Top 5 predictions are 
[[29 23 28 30 25]
 [25 20 18 28 27]
 [ 8  7  4  2  5]
 [17 14  0  1  2]
 [ 5  1  2  7  3]]