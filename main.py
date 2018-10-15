from perceptron import *
from NeuralNetFunctions import *
from DataManipulationFunctions import *
import sys
import numpy as np



if __name__ == "__main__":
    np.seterr(all='raise')
    trainfile = "sonar.arff"
    num_folds = 10
    learning_rate = 0.1
    num_epochs = 10

    data, meta = readData(trainfile)
    X,y = splitData(data)
    folds = stratifyData(X, y, num_folds) # stratified folds


    for i in range(0, num_folds):


        # create test set
        test_X = [X[j] for j in folds[i]]
        test_y = [y[j] for j in folds[i]]
        # create training set
        train_X = [X[j] for j in range(0, len(X)) if j not in folds[i]] # obtain only indices that are not in folds[i]
        train_y = [y[j] for j in range(0, len(X)) if j not in folds[i]]
        train_y  = convertToBinary(train_y, meta)

        # create a neural network
        neural_net = createNeuralNetwork(len(meta.names())-1)

        # train the neural net with the new data
        trainNeuralNetwork(neural_net, train_X, train_y, learning_rate, num_epochs)

        # cross validation
        # iterate through each instance of the test set
        for j in range(len(test_X)):
            confidence = getOutput(test_X[j], neural_net)
            if round(confidence) == 1:
                prediction = meta['Class'][1][1]
            else:
                prediction = meta['Class'][1][0]

            print("Fold = ", i, "| Prediction = ", prediction,"| Actual = ", test_y[j], " | Confidence = ", confidence)


