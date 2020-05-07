import numpy as np
from typing import Tuple, Union


def make_linear(
    samples: int,
    features: int = 1,
    random_seed: Union[int, None] = None,
    noise_std: float = 5.0,
) -> Tuple[np.ndarray, ...]:
    """
    Creates a simple dataset with a noisy linear relationship between X and y

    Parameters
    ----------

    samples : nb of samples/rows
    features : nb of columns/columns
    random_seed : random seed which will be used for all random numbers
    noise_std : standard deviation of the normal noise applied to X and y

    Returns
    ----------
    X : 2d np.ndarray, independent_variable aka regressor/exog
    y : 1d np.ndarray, dependent_variable aka regressand/endog
    weights : 1d np.ndarray of size features+1 which gives the linear relationship before noise. 
        The first value is the intercept/bias
    """

    assert features >= 1 and samples >= 1, "Invalid shape"

    if random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_seed)

    X_noise = rng.normal(scale=noise_std, size=(samples, features))
    y_noise = rng.normal(scale=noise_std, size=(samples, 1))

    slope = rng.uniform(-5, 5, features).reshape(features, -1)
    intercept = rng.normal(0, 20, (1, 1))
    weights = np.vstack([intercept, slope])

    X = np.zeros((samples, features))
    for j in range(features):
        X[:, j] += rng.uniform(-10, 10, samples)
    X += X_noise

    y = X.dot(slope) + intercept + y_noise
    y = y.flatten()
    return X, y.flatten(), weights.flatten()
