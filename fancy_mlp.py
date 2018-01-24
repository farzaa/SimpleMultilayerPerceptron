import numpy as np


def applySigmoid(x, giveMeTheDerivative = False):
	if(giveMeTheDerivative == True):
		return applySigmoid(x) * (1 - applySigmoid(x))
	return 1 / (1 + np.exp(-x))

def print_data(iter, inputs, keys, weights, prediction):
	print "This is iteration # ", iter
	print "Your original input data was... \n", inputs
	print "Your orignal keys were... \n", keys
	print "Your weights at this specific iteration are... \n", weights
	print "Our prediction at this iteration was... \n", prediction
	print "--------------------------------------------------\n"

def train(inputs, keys, layer_one_weights, layer_two_weights):
	for iter in xrange(20000):

		# Layer one will have its own inputs and they are the ones directly given to us from main.
		layer_one_inputs = inputs;

		# Predict just like in simple_mlp.py
		layer_one_prediction = applySigmoid(np.dot(layer_one_inputs, layer_one_weights))

		# Take the prediction from layer one and forward proogate it to the second layer of weights for a final output.
		layer_two_prediction = applySigmoid(np.dot(layer_one_prediction, layer_two_weights))

		# How much were we off by?
		layer_two_error = keys - layer_two_prediction 

		# Change in error just like in simple_mlp.py
		layer_two_change_in_error = layer_two_error * applySigmoid(layer_two_prediction, True)

		# Figure out how wrong our output for layer_one was by seeing how wrong the layer_two_prediction was
		layer_one_error = np.dot(layer_two_change_in_error, layer_two_weights.T)

		# Just like in simple_mlp.py
		layer_one_change_in_error = layer_one_error * applySigmoid(layer_one_error, True)

		# adjust your weights accoridngly.
		layer_one_weights +=  np.dot(layer_one_prediction.T, layer_one_change_in_error)
		layer_two_weights +=  np.dot(layer_two_prediction.T, layer_two_change_in_error)

		if(iter == 0 or iter == 5000 or iter == 9999):
			print_data(iter, inputs, keys, weights, prediction)

	print "Output After Training:"
	print prediction

def main():
	np.random.seed(1)
	inputs = np.array(	[[0,0,1],
						[1,0,1],
						[0,1,1],
						[1,1,1]])

	keys = np.array([[0,1,1,0]]).T
	layer_one_weights = 2*np.random.random((3,4)) - 1
	layer_two_weights = 2*np.random.random((4,1)) - 1
	train(inputs, keys, layer_one_weights, layer_two_weights)

if __name__ == "__main__":
	main()

