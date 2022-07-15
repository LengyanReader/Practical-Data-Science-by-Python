import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm  # Used for solving linear regression problems

def fit_ar(y, p):
    """Fits an AR(p) model. The loss function is the sum of squared errors from t=p+1 to t=n.

    :param y: array (n,), training data points
    :param p: int, AR model order
    :return theta: array (p,), learnt AR coefficients
    """

    # Number of training data points
    n = len(y)
    
    # Construct the regression matrix
    Phi = np.zeros((n-p,p))
    for j in range(p):
        Phi[:,j] = y[p-j-1:n-j-1]
    
    # Drop the first p values from the target vector y
    yy = y[p:]

    # Here we use fit_intercept=False since we do not want to include an intercept term in the AR model
    regr = lm.LinearRegression(fit_intercept=False)
    regr.fit(Phi,yy)
    
    return regr.coef_
    
    
def predict_ar_1step(theta, y_target):
    """Predicts the value y_t for t = p+1, ..., n, for an AR(p) model, based on the data in y_target using
    one-step-ahead prediction.

    :param theta: array (p,), AR coefficients, theta=(a1,a2,...,ap).
    :param y_target: array (n,), the data points used to compute the predictions.
    :return y_pred: array (n-p,), the one-step predictions (\hat y_{p+1}, ...., \hat y_n) 
    """

    n = len(y_target)
    p = len(theta)
    
    # Number of steps in prediction
    m = n-p
    y_pred = np.zeros(m)
    
    for i in range(m):
        t = i+p
        phi = np.flip(y_target[t-p:t]) # (y_{t-1}, ..., y_{t-p})^T
        y_pred[i] = np.sum( phi * theta )
        
    return y_pred
    
    
def plot_history(history, start_at):    
    plt.plot(history.epoch[start_at:], history.history['val_loss'][start_at:])
    plt.plot(history.epoch[start_at:], history.history['loss'][start_at:])
    plt.legend(['Test error','Training error'])    
    plt.xlabel('Epoch')