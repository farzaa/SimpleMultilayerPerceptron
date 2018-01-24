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

def train(inputs, keys, weights):
	for iter in xrange(20000):
		prediction = applySigmoid(np.dot(inputs, weights))
		error = keys - prediction
		change_in_error = error * applySigmoid(prediction,True)
		weights += np.dot(inputs.T ,change_in_error)
		if(iter == 0 or iter == 5000 or iter == 9999):
			print_data(iter, inputs, keys, weights, prediction)

	print "Output After Training:"
	print prediction

def main():
	np.random.seed(1)
	inputs = np.array(	[[0,0,1],
						[1,1,1],
						[1,0,1],
						[0,1,1]])

	keys = np.array([[0,1,1,0]]).T
	weights = 2*np.random.random((3,1)) - 1
	train(inputs, keys, weights)

if __name__ == "__main__":
	main()

