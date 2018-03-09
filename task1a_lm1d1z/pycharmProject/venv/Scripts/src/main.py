import pandas as pd
import numpy as np
import gradientDescent as gd
import regression
import util
import matplotlib.pyplot as plt

myLambda = 100
myLearingRate = 0.1
numIterations = 10
numStepsPerIteration = 10
numCoefs = 10
numSamples = 100

trainPercentage = 0.9

pSrc = np.poly1d([0.9,0.0,-1.9, 0.5, 0.9])
plotXValues = np.arange(-1,1,0.01)





"""
A, YTrain = util.generateDataFromPolynomial(pSrc, -1, 1, numSamples, 0.5)
plt.plot(A, YTrain, 'ro')
plt.plot(plotXValues, pSrc(plotXValues), 'g')


XTrain = [None] * numSamples
for i in range(len(XTrain)):
    XTrain[i] = []
    for j in range(numCoefs):
        XTrain[i] += [pow(A[i], numCoefs - j - 1)]
"""



XComplete, YComplete = util.loadFromCSV("../../../../train.csv", 10, 1)

numSamples = len(XComplete)
numTrainingSamples = int(trainPercentage * numSamples)

XTrain = XComplete[0 : numTrainingSamples, :]
YTrain = YComplete[0 : numTrainingSamples, :]

XTest = XComplete[numTrainingSamples : len(XComplete), :]
YTest = YComplete[numTrainingSamples : len(YComplete), :]

for k in range (10):

    XTrain = XComplete[0: numTrainingSamples, :]
    YTrain = YComplete[0: numTrainingSamples, :]

    XTest = XComplete[numTrainingSamples: len(XComplete), :]
    YTest = YComplete[numTrainingSamples: len(YComplete), :]

    xDataVec = [None] * len(XTrain)

    for i in range(len(xDataVec)):
        xDataVec[i] = np.array(XTrain[i])

    yDataVec = [None] * len(YTrain)
    for i in range(len(yDataVec)):
        yDataVec[i] = YTrain[i]


    ridgeRegressionLossFunction = regression.RidgeLoss(xDataVec, yDataVec, myLambda)

    minimizer = gd.GradientDescent(np.zeros(numCoefs), myLearingRate, ridgeRegressionLossFunction, gd.BoldDriverEvolution(1.1, 0.5))

    print("initial loss")
    print(ridgeRegressionLossFunction.evalLoss(minimizer.w))
    print("\n")
    for i in range(numIterations):
        minimizer.step(numStepsPerIteration)
        print(ridgeRegressionLossFunction.evalLoss(minimizer.w))
        grad = ridgeRegressionLossFunction.evalGradient(minimizer.w)
        #print(grad.transpose().dot(grad))
        print("\n")

    print("final loss")
    print(ridgeRegressionLossFunction.evalLoss(minimizer.w))
    print("\n")

    print("final w")
    print(minimizer.w)
    print("\n")


    yFit = [None] * len(YTest)
    for i in range(len(yFit)):
        yFit[i] = minimizer.w.transpose().dot(XTest[i])

    print("final RMS error")
    print(regression.RMSE(yFit, YTest))


"""


pFit = np.poly1d(minimizer.w)
plt.plot(plotXValues, pFit(plotXValues), 'b')


plt.show()
"""








