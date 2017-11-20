from numpy import array, exp, random, dot

class NeuralNetwork:

    def __init__(self):
        self.synaptic_weights = 2*random.random((3, 1))-1

    # Sigmoid
    def __sigmoid__(self,x):
        return 1/(1+exp(-x))

    # Sigmoid rate of change
    def __sigmoid_derivative__(self, x):
        return x*(1 - x)

    def think(self,inputs):
        output_from_layer1 = self.__sigmoid__(dot(inputs, self.synaptic_weights))
        return output_from_layer1

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for iteration in range(number_of_iterations):
            output_from_layer1 = self.think(training_set_inputs)
            error = training_set_outputs - output_from_layer1
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative__(output_from_layer1))
            self.synaptic_weights +=adjustment;


if __name__ == "__main__":
    # Training inputs and outputs
    training_set_inputs = array([[1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1]])
    training_set_outputs = array([[1, 0, 0, 1, 0]]).T
    layer1 = NeuralNetwork()

    print ("\nRANDOM STARTING WEIGHTS : ")
    print (layer1.synaptic_weights)

    layer1.train(training_set_inputs,training_set_outputs,10000)

    print("\nTrained Weights")
    print (layer1.synaptic_weights)

    print("\nNew inputs")
    print(layer1.think(array([1,1,1])))






