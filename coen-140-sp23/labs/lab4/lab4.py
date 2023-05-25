import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
from sklearn.datasets import fetch_california_housing

hd = fetch_california_housing()
housing = pd.DataFrame(hd.data, columns=hd.feature_names)
housing['target'] = hd.target

def gradient(t, X, y):
    """Compute the current error and gradient."""
    # Hypothesis/estimate values for y
    # Hypothesis: h(x) = 0^T*x
    y_estimate = X.dot(t).flatten()
    # Loss - the difference between the estimated and actual values of y
    loss = y.flatten() - y_estimate
    m = float(len(X))
    # Compute gradient
    grad = -(1.0 / m) * loss.dot(X)
    # Cost function value
    cost = (0.5 / m) * np.sum(np.power(loss, 2))
    return grad, cost

def gradient_descent(x1, x2, y, alpha=0.5, tolerance=1e-5, maxit=1e+6, nulbias=False):
    """Finds the best line fit for predicting y given x.
    Keep track of and also return tested models, gradients, and errors
    along the optimization path.
    """
    # add intercept term to x -- acounts for the bias -- and normalize x's
    # 
    # np.vstack takes 0 = [1, 1, 1, ... 1] x = [x1, x2, x3, ... xn], v = [v1, v2, v3, ... vn] 
    # and converts it to 
    # X = [ [1 x1 v1]
    #       [1 x2 v2]
    #       [1 x3 v3]
    #        ...
    #       [1 xn vn] ]
    # np.ones_like() returns array/matrix of 1s that is the same shape as the input
    X = np.vstack((np.ones_like(x1), x1/x1.max(), x2/x2.max())).T
    # start with a random (or zeros) theta vector, 3 since there are two dependent variables
    t = np.random.randn(3)
    if nulbias:
        t[0] = 0
    # perform gradient descent
    it = 0
    models = []
    grads = []
    errors = []
    while it < maxit:
        grad, error = gradient(t, X, y)
        models.append(t)
        grads.append(grad)
        errors.append(error)
        new_t = t - alpha * grad
        if nulbias:
            new_t[0] = 0
        # check whether we should stop
        if np.sum(abs(new_t - t)) < tolerance:
            break
        # update theta
        t = new_t
        it += 1
    if it == maxit:
        print("Warning: reached maximum number of iterations without convergence.")
    return X, t, models, grads, errors

minerror = 1e+6
c = 1
for i in range(len(hd.feature_names)):
    for j in range(i+1,len(hd.feature_names)):
        col1 = hd.feature_names[i]
        col2 = hd.feature_names[j]
        data = housing.loc[:,[col1,col2,'target']].to_numpy()
        x1 = data[:,0]
        x2 = data[:,1]
        y = data[:,2]
        X, t, models, grads, errors = gradient_descent(x1, x2, y)
        y_estimate = X.dot(t).flatten()
        mse = mean_squared_error(y,y_estimate)
        rmse = math.sqrt(mse)
        
        if rmse < minerror:
            minerror = rmse
            features = (col1,col2)

        print(f"{c}: [{col1},{col2}] RMSE:{rmse}, MSE: {mse}")
        c += 1

print('\n\nMinimum error:')
print(features, minerror)