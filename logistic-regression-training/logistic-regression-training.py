import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for i in range(steps):
        z = X @ w + b
        y_hat = _sigmoid(z)

        dw = (1 / N) * (X.T @ (y_hat - y))
        db = (1 / N) * np.sum(y_hat - y)

        w -= lr * dw
        b -= lr * db
    
    return w, b
    