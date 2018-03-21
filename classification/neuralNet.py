import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x) 

in_neurons = 20
l1_neurons = 12
l2_neurons = 6
out_neurons = 5
samplesize = 100	#arbitrarily chosen
learningRate = 0.1 	#arbitrarily chosen


np.random.seed(1)

X = np.random.randint(2, size=(samplesize, in_neurons))
Y = np.random.randint(2, size=(samplesize, out_neurons))

w0 = 2 * np.random.random((in_neurons, l1_neurons)) - 1
w1 = 2 * np.random.random((l1_neurons, l2_neurons)) - 1
w2 = 2 * np.random.random((l2_neurons, out_neurons)) - 1

print(X)
print ("\n\n")
print(w0)
print ("\n\n")
print(w1)
print ("\n\n")
print(w2)
print ("\n\n")
print(Y)
for i in range(0, 1000):	#chosen arbitrarily
	l0 = X
	l1 = sigmoid(np.dot(l0, w0))
	l2 = sigmoid(np.dot(l1, w1))
	l3 = sigmoid(np.dot(l2, w2))

	l3_error = Y - l3 								#error in output
	l3_delta = l3_error * sigmoid_derivative(l3)	#delta for output layer, will be used to alter the weights

	l2_error = np.dot(l3_delta, w2.T)
	l2_delta = l2_error * sigmoid_derivative(l2)

	l1_error = np.dot(l2_delta, w1.T)
	l1_delta = l1_error * sigmoid_derivative(l1)

	w2 += np.dot(l2.T, l3_delta) * learningRate
	w1 += np.dot(l1.T, l2_delta) * learningRate
	w0 += np.dot(l0.T, l1_delta) * learningRate

print("AFTER FOR LOOP:\n\n")
print(l0)
print ("\n\n")
print(l1)
print ("\n\n")
print(l2)
print ("\n\n")
print(l3)
print ("\n\n")
print("percentage error (final) "+str(np.mean(l3_error)))