
import matplotlib.pyplot as plt
import collections


def plotROC(actual_vector, predicted_vector, confidences_vector):
    # we assume the true class is mine.
    # gather how many instances in the actual vector have mine

    # print(actual_vector)
    # print(predicted_vector)
    # print(confidences_vector)

    counts = dict(collections.Counter(actual_vector))
    # print(counts)
    # sort based on confidences
    sorted_confidence_indices = [x[0] for x in sorted(enumerate(confidences_vector), key=lambda x:x[1])]
    sorted_confidence_indices.reverse()
    # print(sorted_confidence_indices)
    new_confidences = []
    new_labels = []
    for s in sorted_confidence_indices:
        new_confidences.append(confidences_vector[s])
        new_labels.append(actual_vector[s])
    #
    # print(new_confidences)
    # print(new_labels)
    TP = 0
    FP = 0
    last_TP = 0
    TPR_list = []
    FPR_list = []
    for i in range(len(new_labels)):
        if i > 0:
            # print('Check1')
            if (new_confidences[i]!=new_confidences[i-1]):
                # print("Check2")
                if new_labels[i] == "Rock":
                    # print("Check3")
                    if TP > last_TP:
                        # print("Check4")
                        FPR = 1.0* FP/counts["Rock"]
                        TPR = 1.0 * TP/counts["Mine"]

                        TPR_list.append(TPR)
                        FPR_list.append(FPR)
                        last_TP = TP

        if new_labels[i] == 'Mine':
            TP+=1
        else:
            FP+=1

    FPR = FP / counts["Rock"]
    TPR = TP / counts["Mine"]
    TPR_list.append(TPR)
    FPR_list.append(FPR)
    plt.plot(FPR_list, TPR_list)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


def plot_num_epochs_accuracy():
    num_epochs = [25,50,75,100]
    accuracy = [0.8221153846153846, 0.8221153846153846, 0.8125,0.8221153846153846]
    plt.plot(num_epochs, accuracy, 'ro')
    plt.ylim((0,1))
    plt.xticks(num_epochs)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def plot_num_folds_accuracy():
    num_folds = [5,10,15,20,25]
    accuracy = [0.8269230769230769, 0.8028846153846154, 0.8413461538461539, 0.8509615384615384, 0.8173076923076923]
    plt.plot(num_folds, accuracy, 'ro')
    plt.ylim((0,1))
    plt.xticks(num_folds)
    plt.xlabel("Folds")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    plot_num_folds_accuracy()
