import abc

class GradientDescent:

    def __init__(self, w0, initialLearingRate, lossFunction, learningRateEvolutionFunction):
        self.w = w0
        self.learningRate = initialLearingRate
        self.lossFunction = lossFunction
        self.learningRateEvolutionFunction = learningRateEvolutionFunction
        self.coefHistory = [w0]


    def step(self, n = 1):
        for i in range(n):
            self.w = self.w - self.learningRate * self.lossFunction.evalGradient(self.w)
            self.coefHistory += [self.w]
            self.learningRate = self.learningRateEvolutionFunction.evolve(self.learningRate, self.lossFunction, self.coefHistory)




class LearningRateEvolutionFunction:

    @abc.abstractmethod
    def evolve(self, currentLearningRate, lossFunction, coefHistory):
        return


class BoldDriverEvolution(LearningRateEvolutionFunction):

    def __init__(self, acceleration, deceleration):
        self.acceleration = acceleration
        self.deceleration = deceleration

    def evolve(self, currentLearningRate, lossFunction, coefHistory):

        currentLoss = lossFunction.evalLoss(coefHistory[len(coefHistory) - 1])
        lastLoss = lossFunction.evalLoss(coefHistory[len(coefHistory) - 2])

        if (currentLoss < lastLoss):
            return currentLearningRate * self.acceleration
        else:
            return currentLearningRate * self.deceleration





def boldDriver(currentLearningRate, lossFunction, coefHistory):
    currentLoss = lossFunction.evalLoss(coefHistory[len(coefHistory)-1])
    lastLoss = lossFunction.evalLoss(coefHistory[len(coefHistory) - 2])

    if (currentLoss < lastLoss):
        return currentLearningRate * 1.1
    else:
        return currentLearningRate * 0.5


