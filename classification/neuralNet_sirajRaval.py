import numpy as np
##some form of class or smth needs to be added before the neural network for information collection

#defining sizes of layers (number of connections for each neuron in each layer)
IN_neurons = 8#20
L1_neurons = 6#12
L2_neurons = 4#6
OUT_neurons = 2#5

##sample size will depend on the number of test cases, thus will have to be incorporated as a form of counter when reading images. (nahseb)
##assuming static value for now
samples = 10#100

np.random.seed(1)
IN_L1_weights = 2 * np.random.random((IN_neurons, L1_neurons)) - 1
L1_L2_weights = 2 * np.random.random((L1_neurons, L2_neurons)) - 1
L2_OUT_weights = 2 * np.random.random((L2_neurons, OUT_neurons)) - 1



class NeuralNetwork():
    #def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        #np.random.seed(1)
        #IN_L1_weights = 2 * np.random.random((IN_neurons, L1_neurons)) - 1
        #L1_L2_weights = 2 * np.random.random((L1_neurons, L2_neurons)) - 1
        #L2_OUT_weights = 2 * np.random.random((L2_neurons, OUT_neurons)) - 1
        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        ##self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        #for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            ##output = self.think(training_set_inputs)

            #--------
            #each layer requires the same thinking process, just it is fed the output of the previous layer
        L1 = self.think(training_set_inputs, IN_L1_weights)
        print("\nweights for in-layer1 part:\n")
        print(IN_L1_weights)
        print("output for layer 1 == input for layer 2\n")
        print(L1)
        L2 = self.think(L1, L1_L2_weights)
        print("\nweights for layer1-layer2 part:\n")
        print(L1_L2_weights)
        print("\n\noutput for layer 2 == input for output layer\n")
        print(L2)
        OUT = self.think(L2, L2_OUT_weights)
        print("\nweights for layer2-out part:\n")
        print(L2_OUT_weights)
        print("\n\nresulting output")
        print(OUT)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            ##error = training_set_outputs - output
            #--------
            #for multi layer it is more complex, let's just leave it out for now

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            #adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            #self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs, weights):
        # Pass inputs through our neural network (our single neuron).
        #return self.__sigmoid(np.dot(inputs, weights))
        ##What we want is for the matrix to be returned, every time, thus do not perform the sigmoid. 
        return self._sigmoid(np.dot(inputs, weights))

if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    #print "Random starting synaptic weights: "
    #print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    #training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    #training_set_outputs = np.array([[0, 1, 1, 0]]).T

    ##for testing purposes, generating a random binary matrix for inputs and outputs, seeded to see what happens
    IN = np.random.randint(2, size=(samples, IN_neurons))
    print("INPUTS TO NETWORK:\n")
    print (IN)
    OUT_expected = np.random.randint(2, size=(samples, OUT_neurons))
    print("\n\nexpected outputs:\n")
    print(OUT_expected)
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(IN, OUT_expected, 10000)

    #print "New synaptic weights after training: "
    #print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    #print "Considering new situation [1, 0, 0] -> ?: "
    #print neural_network.think(np.array([1, 0, 0]))
