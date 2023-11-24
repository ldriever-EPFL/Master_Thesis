import numpy as np

'''
To be more reflective of the true use case, the function is defined in a seperate file

As test case we use a simple exponential function, which keeps the cost of the test low
'''


def expensive_func(x1, x2, x3, x4, x5):

    return x1 * np.exp(- np.sqrt(x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2))
