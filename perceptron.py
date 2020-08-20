import util
PRINT = True

class PerceptronClassifier:

    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()


    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
      self.features = trainingData[0].keys()
      tempWeights = util.Counter()
      for itr in range(self.max_iterations):
        for i in range(len(trainingData)):
          for l in self.legalLabels:
            tempWeights[l] = trainingData[i].__mul__(self.weights[l])
          if not(trainingLabels[i] == tempWeights.argMax()):
            self.weights[trainingLabels[i]].__radd__(trainingData[i])
            self.weights[tempWeights.argMax()].__sub__(trainingData[i])			


    def classify(self, data ):
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


