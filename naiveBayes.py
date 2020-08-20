import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):

        bestAccuracyCount = -1

        priorProbs = util.Counter()
        conditionalProbs = util.Counter() 
        labelCounts = util.Counter() 

        bestPriorProbs = util.Counter()
        bestConditionalProbs = util.Counter()
        bestK = -1

        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = trainingLabels[i]
            priorProbs[label] += 1
            for feat, value in datum.items():
                labelCounts[(feat,label)] += 1
                if value > 0: 
                    conditionalProbs[(feat, label)] += 1

        for k in kgrid: 
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()

            for key, val in priorProbs.items():
                prior[key] += val

            for key, val in labelCounts.items():
                counts[key] += val

            for key, val in conditionalProbs.items():
                conditionalProb[key] += val

            for label in self.legalLabels:
                for feat in self.features:
                    conditionalProb[(feat, label)] += k
                    counts[(feat, label)] += 2 * k 

            prior.normalize()
            for x, count in conditionalProb.items():
                conditionalProb[x] = count * 1.0 / counts[x]

            self.prior = prior
            self.conditionalProb = conditionalProb

            predictions = self.classify(validationData)
            correctPredictions = 0
            for i in range(len(validationLabels)):
              if predictions[i] == validationLabels[i]:
                correctPredictions += 1 

            if correctPredictions > bestAccuracyCount:
                bestPriorProbs = prior
                bestConditionalProbs = conditionalProb
                bestK = k
                bestAccuracyCount = correctPredictions

        self.prior = bestPriorProbs
        self.conditionalProb = bestConditionalProbs
        self.k = bestK

    def classify(self, testData):
        predictions = []
        self.posteriorProbs = [] 
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            predictions.append(posterior.argMax())
            self.posteriorProbs.append(posterior)
        return predictions

    def calculateLogJointProbabilities(self, datum):

        logJoint = util.Counter()

        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior[label])

            for feat, value in datum.items():
                if value > 0:
                    logJoint[label] += math.log(self.conditionalProb[feat,label])

                else:
                    logJoint[label] += math.log(1-self.conditionalProb[feat,label])

        return logJoint


