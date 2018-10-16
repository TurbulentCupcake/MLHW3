from perceptron import *
from NeuralNetFunctions import *
from DataManipulationFunctions import *
import sys
import numpy as np
from partb import *
import collections

if __name__ == "__main__":
    np.seterr(all='raise')
    trainfile = sys.argv[1]
    num_folds = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_epochs = int(sys.argv[4])

    data, meta = readData(trainfile)
    X,y = splitData(data)
    folds = stratifyData(X, y, num_folds) # stratified folds
    # print(folds)
    confidence_vector = []
    actual_vector = []
    prediction_vector = []
    index_vector = []
    fold_vector = []


    for i in range(0, num_folds):
        print('Fold = ', i )
        for index in folds[i]:
            index_vector.append(index)

        # create test set
        test_X = [X[j] for j in folds[i]]
        test_y = [y[j] for j in folds[i]]
        # create training set
        train_X = [X[j] for j in range(0, len(X)) if j not in folds[i]] # obtain only indices that are not in folds[i]
        train_y = [y[j] for j in range(0, len(X)) if j not in folds[i]]
        train_y = convertToBinary(train_y, meta)

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

            actual_vector.append(test_y[j])
            prediction_vector.append(prediction)
            confidence_vector.append(confidence)
            fold_vector.append(i)
    # print data
    new_fold_output, new_prediction_output, new_actual_output, new_confidence_output = printPredictions(index_vector, fold_vector, actual_vector, prediction_vector, confidence_vector)
    # plotROC(new_actual_output, new_prediction_output, new_confidence_output)
    hits = []
    for i, j in zip(new_prediction_output, new_actual_output):
        hits.append(i==j)

    counts = dict(collections.Counter(hits))
    print('Accuracy for folds=', num_folds,' num_epochs = ',num_epochs, ' is ', counts[True]/len(hits))




