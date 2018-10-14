from perceptron import *
from NeuralNetFunctions import *
from DataManipulationFunctions import *
import sys

if __name__ == "__main__":

    trainfile = "sonar.arff"
    num_folds = 10
    learning_rate = 1
    num_epochs = 10

    data, meta = readData(trainfile)
    X,y = splitData(data)
    folds = stratifyData(X, y, num_folds) # stratified folds

    print(folds)
    neural_net = createNeuralNetwork(len(meta.names())-1)
    # print(getOutput(X[0], neural_net))