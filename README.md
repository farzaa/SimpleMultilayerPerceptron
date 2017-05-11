# SimpleMultilayerPerceptron by Farzain M (call me Farza)

Simple example to understand how the most basic neural network is actually working. To run it:
```
pip install numpy
python simple.py
```

Lets look at the train method which is where all the magic happens.
```python
def train(inputs, keys, weights):
  for iter in xrange(10000):
	  prediction = applySigmoid(np.dot(inputs, weights))
	  error = keys - prediction
	  change_in_error = error * applySigmoid(prediction,True)
	  weights += np.dot(inputs.T ,change_in_error)
```

We'll look at the number of iterations ```iter``` later.

The first thing we do is calculate our prediction ```np.dot(inputs, weights)``` which is basically forward propagation where we dot the inputs by the weights at that current iteration. What exactly are the weights doing?

We want the weights to account for the relationship between the input and the output. For example, imagine I was training a neural network that helped me figure out what foods I liked best. I'd train the network by giving it a bunch of foods I liked  (ex. ice cream, pizza, orange chicken) and didn't like (peas, carrots, etc). How does the network actually "learn" what foods I like and don't like? By adjusting the weights and understanding the RELATIONSHIP between the attributes (ex. ingredients, color, etc) of each food and whether or not I like them.

Let's say a specific (single) neuron in my network focused on if I would like a food based on how sweet that food was. I train this by inputting all the stuff I eat. The weights are adjusted so that food that is more sweet has an output of "LIKE" and food that is not sweet has an output of "DISLIKE". The weights are adjusted every single iteration of training by some amount, slowly it is this weight matrix that gets "smarter" and has the ability to make decisions on its own because it understands the function between food, sweetness, and LIKE or DISLIKE

Side note: This isn't completely right because I like pizza, but pizza isn't sweet! Though, remember this is just a single neuron in our network. Imagine our network outputs a final "LIKE" or "DISLIKE" based on all the outputs of all the different nodes (ex. a node for savory food, a node for green colored food, a node for oily food).

Let's get more technical. Back to the code.

```
prediction = applySigmoid(np.dot(inputs, weights))
```

At iteration 0, our weights are initialized to some random values (more on this later) and haven't been trained yet which means are prediction will most likely be very wrong. But lets see it for ourselves. 

The first iteration ```iter = 0```.
Our inputs are dotted with the weights, remember these are initially random! A 4x3 matrix dotted with a 3x1 matrix is proper input and will give us a 4x1 matrix which is exactly right since the size of keys is 4x1 as well.
```
[[0 0 1]            [[-0.16595599]
 [1 1 1]    dot      [ 0.44064899] 
 [1 0 1]             [-0.99977125]]
 [0 1 1]]
 ```
 
 This 4x1 matrix result is our prediction, and we get:
 ```
[[-0.99977125]
 [-0.72507825]
 [-1.16572724]
 [-0.55912226]]
 ```
 
 Remember our keys are:
 ```
 [[0]
 [1]
 [1]
 [0]]
 ```
 
Our prediction is WAY off. But that is to be expected at iteration 0 because our system hasn't been trained at all. It just picked random weights and hoped for the best. Once we find this dot product we do applySigmoid() on it. This is the activation function and this specific activation function introduces nonlinearity into our model. Let's break this down. 

An activation function (from Wikipedia) "defines the output of that node given an input or set of inputs". When we do ```np.dot(inputs, weights)```, this gives us a combination based on the LINEAR relationship of the two matrixes. Let's see why we don't necessarily want linearity in our models. 

Look at this situation below. Imagine the black dots are junk foods I love/hate sometimes and white dots are vegetables I love/hate sometimes and this is the output of a network that decided how much I liked these two types of foods. How would you draw a line separating these two classes from each other? 

![XOR](http://csci431.artifice.cc/images/xor-plot.png)

Try as much as you can and you'll find that there is no solution by simply drawing a line. This means the issue of deciding what foods I like is a NONLINEAR problem.

This output above actually comes from the XOR truth table, and the way we separate the two classes is via a nonlinear solution.

![XOR2](http://www.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/img43.gif)

Imagine if you had a MASSIVE neural network with many layers and nodes that decided my favorite foods. Now imagine this net didn't utilize a nonlinear activation function like Sigmoid or something else. This means our big, complex neural network is only as good as a single layer perceptron! WTF WHY?? This is because summing up a bunch of linear functions is going to just give us another linear model and we get the same problem as the XOR issue above. We need nonlinearity in the model to avoid this issue of our big, fancy neural net boiling down to a single layer perceptron.
 
This is where the Sigmoid function comes in.

![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

It is a differentiable (important later) nonlinear function and it will squash any number (from - infinity to infinity) to a number between 0 and 1.

So here the line once again
```
prediction = applySigmoid(np.dot(inputs, weights))
```

And our output at iteration 0 is going to be. Notice how every number has been "squashed" between 0 and 1.
```
[[ 0.2689864 ]
 [ 0.3262757 ]
 [ 0.23762817]
 [ 0.36375058]]
```

In summary, we are using forward propagation using our weights and inputs to predict the right answer and we are usign the activation function to introduce nonlinearity.


On to the next line
```
error = keys - prediction
```

Here we are, at the simplest level, we're calculating loss by subtracting the right answers from our predictions and saving that difference. This essentially says "By how much was I off from the right answer?". We use this to adjust our weights later. Lets talk about this a little more before getting to the next part.

![SigmoidSlopes](http://iamtrask.github.io/img/sigmoid-deriv-2.png)

Above is the Sigmoid fucntion again, but this time the slopes are highlighted. You'll notice that the slopes near the top and bottom are relatively shallow but the slopes near the center are much steeper. We can calculate these exact slopes by finding the derivative of the Sigmoid fucntion and plugging in a value for x. These slopes essentially tell us how right or wrong we were. 

If we were WRONG, than the slope will be like the slope in the center and we'll change our weights by some factor of this slope. 

If we CLOSE TO BEING CORRECT, than the slope will be shallow and we'll STILL change our weights but it will change by some much smaller factor since our slope is much smaller. 

This makese sense because we want to change our weights by a larger amount if we were wrong and a smaller amount if were close to right. 

Something to keep in mind is that the neural network will never be exactly right, but we can get pretty damn close. 
 
 


 


