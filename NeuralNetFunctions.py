from perceptron import Perceptron
import random
import math
from DataManipulationFunctions import *
import numpy as np
np.seterr("")
def createNeuralNetwork(numInputs):
    """
    Creates a fully connected neural network with one hidden layer (with
    same number of nodes as input) and one output layer (with one output
    node)
    :param numInputs: integer
    :return: list containing nodes for each level
    """
    # random.seed(1)
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
        # print(p.weights)
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
        h.nextNodes = output_node

    # for the output node, link to all hidden nodes
    output_node.prevNodes = hidden_nodes

    return [input_nodes, hidden_nodes, output_node]

def sigmoid(x):

    return 1.0/(1.0 + math.exp(-x))

def derivative_sigmoid(x):

    x = np.array(x, dtype=np.longfloat)
    out = x * (1 - x)
    return out


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
            total += neural_net[1][i].weights[j]*input[j-1]
        total += neural_net[1][i].weights[0]
        neural_net[1][i].outputValue = sigmoid(total)  # feed through sigmoid

    # Step 3: compute the output value for the output layer
    total = 0
    for i in range(1, len(neural_net[2].weights)):
        total += neural_net[2].weights[i]*neural_net[2].prevNodes[i-1].outputValue
    total += neural_net[2].weights[0]
    neural_net[2].outputValue = sigmoid(total)

    return neural_net[2].outputValue



def derivative_cross_entropy(output, y):
    print('y ', y , 'output ', output)
    if y == 0:
        return 1/(1-output)
    elif y == 1:
        return -1/y

def cross_entropy_error(output, y):
    return -y*math.log(output) - (1-y)*math.log(1-output)

def trainNeuralNetwork(neural_net, train_X, train_y, l_r, num_epochs):
    # warnings.filterwarnings('error')
    # number of epochs means how many times you go over the dataset
    for i in range(num_epochs):
        j = 0
        print('Epoch ', i)
        idx2 = list(range(len(train_X)))
        random.shuffle(idx2)
        for i in idx2:
            output = getOutput(train_X[i], neural_net)
            # print(y, output)
            backpropogate(neural_net, train_X[i], train_y[i], output,  l_r)
            j+=1
        print("Weights for output")

        print(neural_net[2].weights)
        print("Hidden layer nodes weights")
        for i in range(0,2):
            print(neural_net[1][i].weights)


def backpropogate(neural_net, X, y, output,  l_r):



    # compute delta for the output node
    # The explanation for this is found at https://www.ics.uci.edu/~pjsadows/notes.pdf (first 2 pages)
    delta_output = output - y

    # compute the dE/dw for each weight that connect
    # the hidden layer to output
    diff_weights_output = []
    # append delta_output, as this is simply the change for bias
    diff_weights_output.append(delta_output)

    # for each of the hidden node values, multiply them with the delta of the output
    for i in range(len(neural_net[2].prevNodes)):
        diff_weights_output.append(delta_output * neural_net[2].prevNodes[i].outputValue)



    # Now, compute the deltas for all the hidden layer nodes
    # Typically, we would need to sum up the product of the delta of the output and the weight
    # connecting the hidden to the output. But here because there is only one output, we can
    # simply multiply the delta and weight of the link from hidden to output and then
    # multiply this with the derivative_sigmoid of z_output for each node

    z_derivatives_hidden = []

    # for each hidden layer
    for i in range(len(neural_net[1])):
        # compute the z_value

        z_value = derivative_sigmoid(neural_net[1][i].outputValue)
        z_derivatives_hidden.append(z_value)

    # now, compute deltas for each hidden layer
    delta_hidden = []

    for i in range(len(z_derivatives_hidden)):
        d = neural_net[1][i].nextNodes.weights[i+1] * delta_output * z_derivatives_hidden[i]
        delta_hidden.append(d)

    # now we have all the delta outputs for the hidden layer
    # now we must compute the delta_hidden*xi for every input xi,
    # using this computed value, we will modify the weights that connect
    # the hidden layer node to the input layer nodes

    assert (len(diff_weights_output) == len(neural_net[2].weights))
    for i in range(len(diff_weights_output)):
        neural_net[2].weights[i] = neural_net[2].weights[i] - (l_r * diff_weights_output[i])

    # iterate through each hidden node
    for i in range(len(neural_net[1])):

        # iterate through the prev nodes of each hidden node
        for j in range(len(neural_net[1][i].prevNodes)):

            # subtract from the existing weight between this prev node
            # and the current hidden node, the computed delta for that
            # hidden node times the value of the input node
            # times the learning rate.
            neural_net[1][i].weights[j+1] = neural_net[1][i].weights[j+1] -\
                                            (l_r * delta_hidden[i] * X[j])

        # add correction to the bias
        neural_net[1][i].weights[0] = neural_net[1][i].weights[0] - (l_r*delta_hidden[i])





if __name__ == '__main__':

    data, meta = readData("sonar.arff")
    X,y = splitData(data)
    neural_net = createNeuralNetwork(len(meta.names())-1)
    print(getOutput(X[0], neural_net))