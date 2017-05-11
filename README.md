# SimpleMultilayerPerceptron

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

The first thing we do is calculate our prediction ```np.dot(inputs, weights)``` which is basically forward propogation where we dot the inputs by the weights at that current interation. What exactly are the weights doing?

We want the weights to account for the realtionship between the input and the output. For example, imagine I was training a neural network that helped me figure out what foods I liked best. I'd train the network by giving it a bunch of foods I liked  (ex. icecream, pizza, orange chicken) and didn't like (peas, carrots, etc). How does the network actually "learn" what foods I like and don't like? By adjusting the weights and understanding the RELATIONSHIP between the attributes (ex. ingredients, color, etc) of each food and whether or not I like them.

Lets say a specific (single) neuron in my network focused on if I would like a food based on how sweet that food was. I train this by inputting all the stuff I eat. The weights are adjusted so that food that is more sweet has an output of "LIKE" and food that is not sweet has an output of "DISLIKE". The weights are adjusted every single iteration of training by some amount, slowly it is this weight matrix that gets "smarter" and has the ability to make decisons on its own because it understands the function between food, sweetness, and LIKE or DISLIKE

Side note: This isn't completely right because I like pizza, but pizza isn't sweet! Though, remember this is just a single neuron in our network. Imagine our network outputs a final "LIKE" or "DISLIKE" based on all the outputs of all the different nodes (ex. a node for savory food, a node for green colored food, a node for oily food).

Back to our code.

At iteration 0, our weights are initialized to soem random values (more on this later) and haven't been trained yet which means are prediction will most likely be very wrong. But lets test it and go step by step.

Our inputs are dotted without weights. A 4x3 matrix dotted with a 3x1 matrix is proper and will give us a 4x1 matrix.
```
[[0 0 1]            [[-0.04897654]
 [0 1 1]    dot      [ 0.56277106]  
 [1 0 1]             [-0.71071647]]
 [1 1 1]]
 ```
 
 This 4x1 matrix is 
 ```
 [[-0.71071647]
 [-0.14794542]
 [-0.75969301]
 [-0.19692195]]
 ```


 
