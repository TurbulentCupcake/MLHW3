
class Perceptron(object):

    def __init__(self):
        self.isInputNode = True
        self.outputValue = None # for input node, this would just be the input for each node
        self.weights = []
        self.bias = 0
        self.prevNodes = []
        self.nextNodes = []
        self.learningRate = 1

    def setWeights(self, weights):
        pass

    def getOutput(self, input):
        pass


