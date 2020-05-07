
import numpy as np

class LinearRegression:
    """
    A basic linear regression.

    Parameters
    ----------
    method : pseudoinverse/inverse, method of computations of the weights.

    Attributes
    ----------
    weights : 1d np.ndarray containing the weights of the regression. The first one is the bias
    """

    def __init__(self, method:str='pseudoinverse'):
        if method not in ['pseudoinverse', 'inverse']:
            raise NotImplementedError

        self.method = method

    def add_bias(self, X:np.ndarray) -> np.ndarray:
        bias = np.ones((X.shape[0],1))
        return np.hstack([bias,X])

    def fit(self, X:np.ndarray, y:np.ndarray):
        X = self.add_bias(X)

        if self.method == 'pseudoinverse':
            self.weights = np.linalg.pinv(X).dot(y)

        elif self.method == 'inverse':
            self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X:np.ndarray) -> np.ndarray:
        X = self.add_bias(X)
        return X.dot(self.weights)