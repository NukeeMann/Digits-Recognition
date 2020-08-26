# Update
There now two branches.

+ master
  - Where neural network is build using listed below libraries.
+ self-implementation
  - There I have made my own neural network without using any libraries like Lasagne. Everything there was made from scratch including training function, activation functions (tanh and softmax) and defining layers with weights.
  
  
I have set self-implemented branch as default one beacuse it's something that I'm more proud of.

# Digits-Recognition
Simple deep learning project for handwritten digit recognition with feature that allows to quickly creat your own digits.

![Program](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr1.PNG)

## Used libraries
+ Lasagne 
  - for building our neural network
+ Theano and NumPy
  - for mathematical operations
+ OS
  - for MNIST database downloading
+ Pygame
  - for user friendly interface
  
## Database
The database that our network is learning on comes from MNIST which provides us with training set of 60 thousand examples and test set of 10 thousand examples. If files are missing, our program will download them automatically. Then we reshape the data to 28x28 pixels format wich will represent our digits. To every sample there is assigned label that tells which number it is.

## Neural network
Our neural network is made of 4 layers - Input layer, 2 hidden layers and output layer.
Input layer has 784 nodes - 28x28 pixels converted to vector of 784 values.
Number of nodes for our two hidden layers are declared in the beginning of our program and we can adjust it.
Output layer has 10 nodes which represent 10 possible digits.

We randomly drop 50% of wages from our hidden layers to avoid overfitting problem.

For activation I choosed nonlinear tahn and softmax functions. 

## Training function
In loop that goes for declared in main function number of epoch we do the following.
Get the prediction of our neural network based on given input. Then we compute cross-entropy between given prediction and actuall labels. Using Nestrov accelerated gradient descending update function we adjust params. For this method I choosed learning rate equal 0.1 and momentum value equal 0.9.

## Results
With 800 nodes and 200 epochs I have achieved accuracy of 93.2 % after about 42 minutes of learing.

I have also created an simple interface wich allows to draw our own digits by coloring squares on 28x28 window. This doesnt work as good as loading actuall picture of handwritten digit because of 
difference between intensity of grey scale by human hand and the one generated by our algorithm.

Despite this results are quite good.

Examples:

![ex_0](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr_0.PNG)

![ex_1](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr_1.PNG)

![ex_2](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr_2.PNG)

![ex_3](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr_3.PNG)

![ex_4](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr_4.PNG)

But not allways. For example number five.

![ex_5](https://github.com/NukeeMann/Digits-Recognition/blob/master/img/dr_5_fail.PNG)
