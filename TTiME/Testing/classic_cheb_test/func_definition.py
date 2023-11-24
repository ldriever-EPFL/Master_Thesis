import numba
import numpy as np

'''
To be more reflective of the true use case, the function is defined in a seperate file

As test case we use a simple exponential function, which keeps the cost of the test low
'''


def expensive_func(x1: float, x2: float, x3: float, x4: float) -> float:

    return x1 * np.exp(- abs(x2)) + x3 ** 2 * np.exp(-abs(x4))
