# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
3. I used the pandas library to calculate summary statistics of the traffic signs data set:
* the size of training set are 34799 photos.
* The size of validation set are 4410 photos.
* The size of test set are 12630 photos.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

After loading the dataset,using matplotlib to visualize the catagories of traffic signs.


![](https://i.imgur.com/2bJ8lJ1.png)
Then I printed a bar chart showing how many examples of each class were available on the training set.

![](https://i.imgur.com/MkY3xZz.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I shuffle the images using shuffle from the sklearn.utils library. X_train, y_train = shuffle(X_train, y_train)
```
# Shuffle the data
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
```

As a last step, I normalized the image data because the normalize can increase the accuarcy of model.

I also tried augmenting the dataset, converting the examples to grayscale and transforming the image but non of this resulted on a better prediction. The normalization was the one that return better results and finally returned a model capable of classifying my own images correctly.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected		| input = 400, output = 120       	            |
| RELU					|												|
| Fully connected		| input = 120, output = 84       	            |
| RELU					|												|
| Fully connected		| input = 84, output = 10       	            |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 20 epochs, a batch size of 128 and a learning rate of 0.001.More eposhs and learning rate will be caused overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95.6%
* test set accuracy of 86%


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  
  Ans:I used a similar architecture to the paper offered by the instructors. I used it because they got such a good score the answer was given through it.
  

* What were some problems with the initial architecture?
  
  Ans:The initial architecture is lack of knowledge of all the parameters.So I change the Input from images with one channel to images with 3 channels and modified the number of labels from 10 to len(all_labels) which in this case was 43.
  
  
* How was the architecture adjusted and why was it adjusted? 
  
  Ans:I do not change the architecture,but tuned some parameters.

* Which parameters were tuned? How were they adjusted and why?
  
  Ans:The epochs,batch size and learning rate are following the some experiments to try the result.For my ephochs,when I get the best accuarcy, I will stop to increase it.learning rate is not truned because I think the 0.001 is good.The training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.
  
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  
  Ans:I think I could go over this project for another week and keep on learning. I think this is a good question and I could still learn more about that. I think the most important thing I learned was having a more uniform dataset along with enough convolutions to capture features will greatly improve speed of training and accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

To test my model is good enough. I find the photos outside the datasets to test.
ex:
![](https://i.imgur.com/O6ZQSrZ.png)

![](https://i.imgur.com/eeu7rmh.png)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry       		| No entry   									|
| Yield     			| Yield 										|
| No entry				| No entry										|
| No entry				| No entry										|
| Stop      			| Stop     		    							|
| Speed limit (70km/h)	| Bumpy road							|
| Keep Right			| Keep right								|

The model was able to correctly predict 5 other 7 traffic signs, which gives an accuracy of 80%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For my images,almost are correct except image6.The model identified as Bumpy road not Speed limit (70km/h).

image1.png:
No entry: 100.00%
Speed limit (30km/h): 0.00%
Stop: 0.00%
Bicycles crossing: 0.00%
No vehicles: 0.00%

image2.png:
Yield: 99.57%
Priority road: 0.43%
No passing for vehicles over 3.5 metric tons: 0.00%
Ahead only: 0.00%
No passing: 0.00%

image3.png:
No entry: 100.00%
Turn left ahead: 0.00%
Bicycles crossing: 0.00%
No vehicles: 0.00%
Stop: 0.00%

image4.jpg:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

image5.png:
Stop: 97.97%
Bicycles crossing: 2.03%
Wild animals crossing: 0.00%
No entry: 0.00%
Speed limit (30km/h): 0.00%

image6.png:
Bumpy road: 100.00%
No entry: 0.00%
Bicycles crossing: 0.00%
Speed limit (70km/h): 0.00%
Speed limit (30km/h): 0.00%

image7.png:
Keep right: 100.00%
Roundabout mandatory: 0.00%
Turn left ahead: 0.00%
No entry: 0.00%
End of no passing: 0.00%

### Summarize the results with a written report
- maybe can using the transfer learning to increase the Accuarcy
