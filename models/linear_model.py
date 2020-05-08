
import numpy as np
import warnings
warnings.simplefilter("always", UserWarning)

class BaseLinearRegression:
    def add_bias(self, X:np.ndarray) -> np.ndarray:
        bias = np.ones((X.shape[0],1))
        return np.hstack([bias,X])

    def predict(self, X:np.ndarray) -> np.ndarray:
        X = self.add_bias(X)
        return X.dot(self.weights)

class ClosedFormLinearRegression(BaseLinearRegression):
    """
    A basic linear regression suing closed form

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

    def fit(self, X:np.ndarray, y:np.ndarray):
        X = self.add_bias(X)

        if self.method == 'pseudoinverse':
            self.weights = np.linalg.pinv(X).dot(y)

        elif self.method == 'inverse':
            self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

class GradientDescentLinearRegression(BaseLinearRegression):
    """
    A linear regression using gradient-descent

    Parameters
    ----------
    method : currently only batch is supported. loss is assumed to be MSE for now
    learning_rate : constant learning rate applied
    tolerance : stopping tolerance applied to the model
    max_iters : max number of iterations during the gradient descent

    Attributes
    ----------
    weights : 1d np.ndarray containing the weights of the regression. The first one is the bias
    """

    def __init__(self, method:str='batch', learning_rate:float=0.1, tolerance:float=1e-3, max_iterations:int=10000):
        if method not in ['batch']:
            raise NotImplementedError
        self.method = method
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, X:np.ndarray, y:np.ndarray):
        X = self.add_bias(X)

        if self.method == 'batch':
            # random initialization
            self.weights = np.random.default_rng().standard_normal(size=X.shape[1])
            has_converged = False
            for i in range(self.max_iterations):
                gradient = 2/X.shape[0] * X.T.dot(X.dot(self.weights)-y)
                self.weights = self.weights - self.learning_rate * gradient
                norm_gradient = np.linalg.norm(gradient)
                if  norm_gradient < self.tolerance:
                    has_converged = True
                    break
            if not has_converged:
                warnings.warn("Max numbers of iterations has been reached")