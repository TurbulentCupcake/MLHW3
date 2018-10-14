import scipy.io.arff as arff
import numpy as np
import pandas as pd
import sys
import os


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