import numpy as np
import abc

# base class for all (linear) loss functions
class LossFunction:

    def __init__(self, xData, yData):
        self.xData = xData
        self.yData = yData

    @abc.abstractmethod
    def evalLoss(self, w):
        return

    @abc.abstractmethod
    def evalGradient(self, w):
        return

# ridge regression loss function
class RidgeLoss(LossFunction):

    def __init__(self, xData, yData, lam):
        LossFunction.__init__(self, xData, yData)
        self.lam = lam

    def evalLoss(self, w):
        n = len(self.xData)
        res = 0
        for i in range(n):
            res += pow(self.yData[i] - w.transpose().dot(self.xData[i]), 2)

        res /= n
        res += self.lam * w.transpose().dot(w)

        return res

    def evalGradient(self, w):
        n = len(self.xData)
        res = 0
        for i in range(n):
            res += (self.yData[i] - w.transpose().dot(self.xData[i])) * self.xData[i]

        res *= -2 / n
        res += self.lam * 2 * w

        return res



# RMS regression loss function
class StandardRegressionLoss(LossFunction):

    def __init__(self, xData, yData):
        LossFunction.__init__(self, xData, yData)

    def evalLoss(self, w):
        n = len(self.xData)
        res = 0
        for i in range(n):
            res += pow(self.yData[i] - w.transpose().dot(self.xData[i]), 2)

        res /= n

        return res

    def evalGradient(self, w):
        n = len(self.xData)
        res = 0
        for i in range(n):
            res += (self.yData[i] - w.transpose().dot(self.xData[i])) * self.xData[i]

        res /= n

        return -2 * res




def RMSE(yTest, yReal):
    n = len(yTest)
    sum = 0
    for i in range(n):
        sum += pow(yTest[i] - yReal[i], 2)

    return (sum / n)**0.5
