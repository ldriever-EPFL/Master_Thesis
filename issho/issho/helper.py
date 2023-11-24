"""
A set of useful helper functions
"""
import numpy as np


def rotate(vec, theta):
    """
    :param vec: the to-be rotated vector (array of shape 2 or 2xn)
    :param theta: rotation angle in radians (array of size n)
    :return: the rotated vector (2xn)
    """
    if len(vec.shape) == 1:
        # Turn into column vector
        vec = vec.reshape(-1, 1)

    if len(vec.shape) > 2 or vec.shape[0] != 2:
        raise Exception("Input vector must be of shape 2xn")

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).reshape((2, 2, -1))

    return np.einsum('ijk,kj->ki', rot_matrix, vec.T).T


def get_vector_angle(vec):
    """
    get the angle (counter-clockwise positive) that a 2D vector makes with the positive x axis
    :param vec: array with two components
    :return: angle in radians
    """

    return np.arctan2(vec[1], vec[0])


def fortran_float(number):
    return '{:.10e}'.format(number).replace('e', 'd')  # 10 decimal places to guarantee good accuracy


def py_float_from_fortran(number_string):
    return float(number_string.replace('d', 'e'))
