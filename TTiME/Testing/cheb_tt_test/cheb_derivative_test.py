import os
import numpy as np
import matplotlib.pyplot as plt
from ttime import preChebTT

import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('macosx')

"""
Tests the implementation of ChebTT.get_derivative()
Make sure to run cheb_test.py first
"""

# Create the preChebTT instance
intervals = np.array([[0, 1],
                     [2, 3],
                     [-1, 3],
                     [1, 4],
                     [0.01, 0.03]])

core_storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cheb_outputs', 'Final_cores')

interp = preChebTT(core_storage_path, intervals)

# Calculate the Chebyshev derivative at a random point in a random direction
np.random.seed(111)
point = np.random.uniform(intervals[:, 0], intervals[:, 1], size=(1, len(intervals)))
direction = np.random.random(len(intervals))
direction *= intervals[:, 1] - intervals[:, 0]   # account for the size of the different intervals
direction /= np.linalg.norm(direction)  # normalize the vector

Cheb_derivative = interp.get_derivative(point)
Cheb_direction_derivative = np.dot(Cheb_derivative, direction)

# Check that specifying certain values for the 'axis' kwarg is consistent with the above result
if interp.get_derivative(point, axis=1) == Cheb_derivative[1]:
    print('passed axis test 1')
else:
    raise Exception('Failed axis test 1')

if np.all(interp.get_derivative(point, axis=(0, 3, 4)) == Cheb_derivative[[0, 3, 4]]):
    print('passed axis test 2')
else:
    raise Exception('Failed axis test 2')

if np.all(interp.get_derivative(point, axis=(0, 1, 2, 3, 4)) == Cheb_derivative):
    print('passed axis test 3')
else:
    raise Exception('Failed axis test 3')

# Calculate the finite difference derivative for increasingly small step sizes at that random point and in that random direction
step_sizes = np.logspace(-1, -10, 1000)
finite_difference_derivatives = []
point_value = interp[point]

for step_size in step_sizes:
    finite_difference_derivatives.append((interp[point + direction * step_size] - point_value) / step_size)

# Look at the convergence of the finite difference derivative relative to the Chebyshev derivative.
# If the error shows first order convergence (i.e. a line of slope 1 on the logplot) until machine precision is reached, the Chebyshev derivative can be considered correct

errors = np.abs(np.array(finite_difference_derivatives) - Cheb_direction_derivative) / np.abs(Cheb_direction_derivative)

plt.title('Convergence Plot of First Order Finite Difference Method')
plt.ylabel('Absolute error')
plt.xlabel('Step size')
plt.loglog(np.logspace(0, -9, 10), np.logspace(0, -9, 10) / 0.1 * errors[0], color='r', linestyle='--', label='reference line of slope 1')
plt.loglog(step_sizes, errors, label='numerical errors', linewidth=2)
plt.legend()
plt.show()

