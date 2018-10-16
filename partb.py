
import matplotlib.pyplot as plt
import collections


def plotROC(actual_vector, predicted_vector, confidences_vector):
    # we assume the true class is mine.
    # gather how many instances in the actual vector have mine

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    TPR = []
    FPR = []
    thresholds = []
    print(predicted_vector == actual_vector)
    hits = []
    for pred, act in zip(predicted_vector, actual_vector):
        hits.append(pred == act)

    counts = dict(collections.Counter(hits))

    for threshold in range(0, 100):

        trueHits = 0
        falseAlarms = 0
        # iterate through the confidence vector checking each confidence
        for i in range(len(confidences_vector)):

            # if the confidence is above the given threshold, then
            if confidences_vector[i] >= threshold/100 :

                # check if it is a match and increment the number of trueHits
                if hits[i] == True:
                    trueHits+=1
                else:
                    falseAlarms+=1

        TPR.append(trueHits/counts[True])
        FPR.append(falseAlarms/counts[False])

    print(TPR)
    print(FPR)
    TPR.reverse()
    FPR.reverse()
    plt.plot(FPR, TPR)
    plt.show()




