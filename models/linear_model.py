import numpy as np
import warnings
from typing import Union, Tuple

warnings.simplefilter("always", UserWarning)


class BaseLinearRegression:
    def add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Add a column of 1 to the left of a ndarray in order to include bias
        """
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get the predictions of a fitted model
        """
        X = self.add_bias(X)
        return X.dot(self.weights)


class ClosedFormLinearRegression(BaseLinearRegression):
    """
    A basic linear regression using closed form

    Parameters
    ----------
    method : pseudoinverse/inverse, method of computations of the weights

    Attributes
    ----------
    weights : 1d np.ndarray containing the weights of the regression. The first one is the bias
    """

    def __init__(self, method: str = "pseudoinverse"):
        if method not in ["pseudoinverse", "inverse"]:
            raise NotImplementedError
        self.method = method

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self.add_bias(X)

        if self.method == "pseudoinverse":
            self.weights = np.linalg.pinv(X).dot(y)

        elif self.method == "inverse":
            self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


class GradientDescentLinearRegression(BaseLinearRegression):
    """
    A linear regression using gradient-descent.
    Loss used during descent is squared-error/mse

    Parameters
    ----------
    batch_size : 
        - "full" is batch gradient descent (computation of the gradient with all data)
        - 1 is stochastic gradient descent
        - int>1 is minibatch with int being the number of samples used for gradient computation
        - 0<float<0 is minibatch with float being the fraction of total samples used
    learning_rate : factor applied to the gradient. 
        Current implemention only supports constants learning rate
    tolerance : stopping criterion, which is the minimum norm of the gradient
    max_iterations : max number of iterations during the gradient descent


    Attributes
    ----------
    weights : 1d np.ndarray containing the weights of the regression. The first one is the bias
    """

    def __init__(
        self,
        batch_size: Union[str, float, int] = "full",
        learning_rate: float = 0.1,
        tolerance: float = 1e-3,
        max_iterations: int = 10000,
        random_seed: Union[int, None] = None,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        if random_seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_seed)

    def compute_gradient(
        self, X: np.ndarray, weights: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient and updates weights according to an MSE loss
        """
        gradient = 2 / X.shape[0] * X.T.dot(X.dot(weights) - y)
        weights -= self.learning_rate * gradient
        return gradient, weights

    def sample_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample rows from (X,y) to adapt to the kind of gradient descent:
            - batch, all-data is used for computing the gradient
            - stochastic, only one sample
            - mini-batch, in between
        """

        if self.batch_size == "full":
            return X, y

        elif isinstance(self.batch_size, int):
            if self.batch_size > X.shape[0]:
                raise ValueError

            random_idxs = self.rng.choice(X.shape[0], self.batch_size)
            return X[random_idxs], y[random_idxs]

        elif isinstance(self.batch_size, float):
            if self.batch_size < 0 or self.batch_size > 1:
                raise ValueError

            true_batch_size = int(X.shape[0] * self.batch_size)
            random_idxs = self.rng.choice(X.shape[0], true_batch_size)
            return X[random_idxs], y[random_idxs]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self.add_bias(X)

        # random initialization
        self.weights = self.rng.standard_normal(size=X.shape[1])
        has_converged = False

        for i in range(self.max_iterations):
            X_sample, y_sample = self.sample_data(X, y)
            gradient, self.weights = self.compute_gradient(
                X_sample, self.weights, y_sample
            )

            norm_gradient = np.linalg.norm(gradient)
            if norm_gradient < self.tolerance:
                has_converged = True
                break

        if not has_converged:
            warnings.warn("Max numbers of iterations has been reached")
