from numpy import array, exp, random, dot
class NeronLayer():
    def __init__(self, number_of_neurons,number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron,number_of_neurons)) -1

class NeuralNetwork:

    def __init__(self,layer1,layer2):
        self.layer1 = layer1
        self.layer2 =layer2

    # Sigmoid
    def __sigmoid__(self,x):
        return 1/(1+exp(-x))

    # Sigmoid rate of change
    def __sigmoid_derivative__(self,x):
        return x*(1 - x)

    def think(self,inputs):
        output_from_layer1 = self.__sigmoid__(dot(inputs,self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid__(dot(output_from_layer1,self.layer2.synaptic_weights))
        return output_from_layer1,output_from_layer2

    def train(self,training_set_inputs,training_set_outputs,number_of_iterations):
        for iteration in range(number_of_iterations):
            output_from_layer1, output_from_layer2 = self.think(training_set_inputs)

            layer2_error = training_set_outputs - output_from_layer2
            layer2_delta = layer2_error * self.__sigmoid_derivative__(output_from_layer2)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error* self.__sigmoid_derivative__(output_from_layer1)

            layer2_adjusment = output_from_layer1.T.dot(layer2_delta)
            layer1_adjusment = training_set_inputs.T.dot(layer1_delta)

            layer1.synaptic_weights +=  layer1_adjusment
            layer2.synaptic_weights -+   layer2_adjusment

            # error = training_set_outputs - output_from_layer1
            # adjustment = dot(training_set_inputs.T,error*self.__sigmoid_derivative__(output_from_layer1))
            # self.synaptic_weights +=adjustment;


if __name__ == "__main__":
    # Training inputs and outputs
    training_set_inputs = array([[1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 0], [0, 0, 1]])
    training_set_outputs = array([[1, 0, 0, 1, 0]]).T
    layer1 = NeronLayer(4,3)
    layer2 = NeronLayer(1,4)

    alice = NeuralNetwork(layer1,layer2)

    print ("RANDOM STARTING WEIGHTS : ")
    print("Layer 1:")
    print (alice.layer1.synaptic_weights)
    print("Layer 2 :")
    print(alice.layer2.synaptic_weights)

    alice.train(training_set_inputs,training_set_outputs,10)

    print("Trained Weights")
    print("LAYER 1")
    print (alice.layer1.synaptic_weights)
    print("LAYER 2")
    print(alice.layer2.synaptic_weights)

    print(alice.layer1.think(array([1,1,1])))



