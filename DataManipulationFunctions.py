import scipy.io.arff as arff
import numpy as np
import pandas as pd
import sys
import os
import random
import collections

def readData(filename):

    if filename == None:
        raise("Filename not provided")
    if not os.path.isfile(filename):
        raise("File not found")

    data, meta = arff.loadarff(filename)

    return data, meta

def splitData(data):
    X = []
    for i in data:
        d = []
        for j in range(0, len(i)-1):
            d.append(i[j])
        X.append(d)

    y = []
    for i in data:
        y.append(i[len(i)-1])

    return X, y

def stratifyData(X, y, num_folds):
    """
    Stratifies data and returns a set of folds that can be iterated over
    :param X:
    :param y:
    :param num_folds:
    :return:
    """

    # get unique class
    classes = list(set(y))
    indices = dict()
    for c in classes:
        indices[c] = []

    # split the indices between the two classes
    for i in range(0, len(y)):
        indices[y[i]].append(i)

    # shuffle both the sets
    for c in classes:
        random.shuffle(indices[c])

    # compute data per fold
    dpf = round(len(X)/num_folds)
    # ratio of classes
    count = dict(collections.Counter(y))
    ratio = count[classes[0]]/len(y)
    # folds will hold the indices for each fold of data
    folds = dict()
    for i in range(num_folds):
        folds[i] = []

    # for each fold
    for i in range(num_folds):

        num_class_0 = round(ratio*dpf)  # number of instances to pull from class 0
        num_class_1 = dpf - num_class_0  # number of instances to pull from class 1

        if num_class_0 <= len(indices[classes[0]]) and num_class_1 <= len(indices[classes[1]]): # ensure both classes have enough data to split
            for j in range(num_class_0):
                folds[i].append(indices[classes[0]].pop())
            for j in range(num_class_1):
                folds[i].append(indices[classes[1]].pop())
        else: # if one of the classes do not have enough data, then simply place all the remaining values into a single fold
            for j in range(len(indices[classes[0]])):
                folds[i].append(indices[classes[0]].pop())
            for j in range(len(indices[classes[1]])):
                folds[i].append(indices[classes[1]].pop())

    return folds


def convertToBinary(y, meta):
    """
    Convert a reponse variable into binary.
    :param y: labels
    :param meta: meta from arff
    :return: y containing binary labels (0 and 1)
    """
    classes = [str.encode(meta['Class'][1][0]), str.encode(meta['Class'][1][0])]
    new_y = []
    for label in y:
        if label == classes[0]:
            new_y.append(0) # if iot is rock
        else:
            new_y.append(1) # if it is mine

    return new_y





