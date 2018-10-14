from perceptron import Perceptron
import random
import math
from DataManipulationFunctions import *

def createNeuralNetwork(numInputs):
    """
    Creates a fully connected neural network with one hidden layer (with
    same number of nodes as input) and one output layer (with one output
    node)
    :param numInputs: integer
    :return: list containing nodes for each level
    """

    neural_net = []
    # Step 1: create the input nodes and store then in an array
    input_nodes = []
    for i in range(0, numInputs):
        input_nodes.append(Perceptron())
    # print(input_nodes)
    # Step 2, create nodes for input layer and randomly assign weights
    hidden_nodes = []
    for i in range(0, numInputs):
        p = Perceptron()
        p.weights = [random.uniform(-1,1) for j in range(0, numInputs+1)] # set random weights
        # note that the random weights assign here correspond to each input node that will be
        # attached later. the weights are IN THE SAME ORDER AS THE INPUT NODES.
        p.isInputNode = False
        hidden_nodes.append(p)
    # print(hidden_nodes)
    # for each hidden layer, have a link back to the input node
    for h in hidden_nodes:
        h.prevNodes = input_nodes

    # for each input layer, have a link forward to the
    for i in input_nodes:
        i.nextNodes = hidden_nodes

    # Step 3, create the output node
    output_node = Perceptron()
    output_node.weights = [random.uniform(-1,1) for j in range(0,numInputs+1)]
    output_node.isInputNode = False
    # print(output_node)

    # for each hidden node, link to the output node
    for h in hidden_nodes:
        h.nextNodes.append(output_node)

    # for the output node, link to all hidden nodes
    output_node.prevNodes = hidden_nodes

    return [input_nodes, hidden_nodes, output_node]


def sigmoid(x):
    return 1/(1+math.exp(-x))

def getOutput(input, neural_net):
    """
    Returns prediction for a given set of inputs
    :param input: a list containing the number of
    :param neural_net:
    :return: a float between 0 and 1 giving the output value
    """

    # Step 1: assign all input values to the input nodes

    for i in range(0,len(input)):
        neural_net[0][i].outputValue = input[i]

    # Step 2: compute the input values for the hidden layer
    for i in range(0, len(input)): # for each hidden layer node
        total = 0
        # compute the input for the hidden node
        for j in range(1,len(neural_net[1][i].weights)): # iterate through each previous node and its corresponding weight
            total += neural_net[1][i].weights[j]*neural_net[1][i].prevNodes[j-1].outputValue
        total += neural_net[1][i].weights[0]
        neural_net[1][i].outputValue = sigmoid(total)  # feed through sigmoid

    # Step 3: compute the output value for the output layer
    total = 0
    for i in range(1, len(neural_net[2].weights)):
        total += neural_net[2].weights[i]*neural_net[2].prevNodes[i-1].outputValue
    total += neural_net[2].weights[0]
    neural_net[2].outputValue = sigmoid(total)

    return neural_net[2].outputValue



def derivative_cross_entropy(neural_net, output, y):
    pass

def trainNeuralNetwork(neural_net, train_X, train_y, l_r, num_epochs):

    # number of epochs means how many times you go over the dataset
    for i in range(num_epochs):
        for X, y in zip(train_X, train_X):
            output = getOutput(X, neural_net)
            error = calculate_cross_entropy_error(neural_net, output, y)
            backpropogate(X, error)




if __name__ == '__main__':

    data, meta = readData("sonar.arff")
    X,y = splitData(data)
    neural_net = createNeuralNetwork(len(meta.names())-1)
    print(getOutput(X[0], neural_net))