#####################
## Necessary: numpy, ipython, matplotlib==2.1.0 (Newer versions broke backward compatibility)
####################

##############
#Official stuff
import numpy as np
import matplotlib.pyplot as plt

############
# Our stuff
import task1a_lm1d1z.pycharmProject.venv.Scripts.src.regression as regression
import task1a_lm1d1z.pycharmProject.venv.Scripts.src.util as util
import task1a_lm1d1z.pycharmProject.venv.Scripts.src.gradientDescent as gd



# Make import from helper classes possible
import sys
sys.path.insert(0, '../lib/helpers')
# Import helper classes
from task1a_lm1d1z.pycharmProject.venv.Scripts.lib.helpers.util import gradient_descent, generate_polynomial_data
import task1a_lm1d1z.pycharmProject.venv.Scripts.lib.helpers.plot_helpers as ph


# Num_points: Number of points the function generates
# noise
def createDataPoints(num_points=100, noise=0.6, slope=3, intercept=1):
    # From Demo Week 1 18F-IML
    w_true = np.array([slope, intercept])
    X, Y = generate_polynomial_data(num_points, noise, w_true)
    return X, Y


XTrain, YTrain = createDataPoints()
xDataVec = util.makeDatapointsToVectors(XTrain)
yDataVec = YTrain

ridgeRegressionLossFunction = regression.RidgeLoss(xDataVec, yDataVec, 0.1)
minimizer = gd.GradientDescent(np.zeros(2), 0.1, ridgeRegressionLossFunction,
                               gd.BoldDriverEvolution(1.1, 0.5))

for i in range(100):
    minimizer.step(10)
    print(ridgeRegressionLossFunction.evalLoss(minimizer.w))
    grad = ridgeRegressionLossFunction.evalGradient(minimizer.w)

print("final loss")
print(ridgeRegressionLossFunction.evalLoss(minimizer.w))
print("\n")

##Plot
fig = plt.subplot(111)
plot_opts = {'x_label': '$x$', 'y_label': '$y$', 'title': 'Closed Form Solution', 'legend': True,
             'y_lim': [np.min(YTrain)-0.5, np.max(YTrain)+0.5]}

ph.plot_data(XTrain[:, 0], YTrain, fig=fig, options=plot_opts)
ph.plot_fit(XTrain, minimizer.w, fig=fig, options=plot_opts)
plt.show()


##Fazit: Funktioniert gut mit genügend kleinen lambda