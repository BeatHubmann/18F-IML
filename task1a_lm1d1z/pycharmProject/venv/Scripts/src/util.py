import pandas as pd
import numpy as np

def loadFromCSV(filename, xDimensions, yDimensions):
    data = pd.read_csv(filename, header=0)
    # Separate training data into Y and X
    array = data.values
    Y = array[:, 1:yDimensions+1]
    X = array[:, (yDimensions+1):(yDimensions+xDimensions+1)]
    return (X,Y)

def generateDataFromPolynomial(p, xStart, xEnd, numPoints, noise):

    xValues = (xEnd - xStart) * np.random.sample(numPoints) + xStart
    xValues = np.sort(xValues)

    noiseData = noise * (np.random.sample(numPoints) - 0.5)
    yData = p(xValues)

    for i in range(numPoints):
        yData[i] += noiseData[i]

    return (xValues, yData)

#Expects: [[x_11, x_12, x_13, ...], [x_21, x_22, x_23], ...]
#Returns: [vec(x_11, x_12, x_13), vec(x_21, x_22, x_23), ...]
def makeDatapointsToVectors(xArray):
    arrayLength = len(xArray)
    xDataVec = [None] * arrayLength

    for i in range(arrayLength):
        xDataVec[i] = np.array(xArray[i])

    return xDataVec
