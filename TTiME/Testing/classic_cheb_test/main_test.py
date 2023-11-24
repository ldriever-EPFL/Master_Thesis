import os
import sys
import numba
import numpy as np
import matplotlib.pyplot as plt
from func_definition import expensive_func
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from ttime import ClassicCheb
import matplotlib
matplotlib.use('MacOSX')

np.random.seed(77)

# Define the inputs
intervals = np.array([[0, 1],
                     [2, 3],
                     [-1, 3],
                     [1, 4]])
order = 10

# # Create the data
# d = len(intervals)
# one_axis_points = [np.cos(np.pi * np.arange(order + 1) / order)] * d
# points = np.array(np.meshgrid(*one_axis_points, indexing='ij')).reshape(d, -1).T
#
# points = (points + 1) / 2 * (intervals[:, 1] - intervals[:, 0]) + intervals[:, 0]  # Map to the actual parameter intervals
#
# vals = expensive_func(*points.T)
#
# # Create the interpolation object
# interp = ClassicCheb(points, vals, intervals, order, input_type='real')
#
# # Test the interpolation for 1000 random points in the specified interval
# n_test_points = 1000
# test_points = np.random.uniform(intervals[:, 0], intervals[:, 1], size=(n_test_points, len(intervals)))
#
# np.savetxt('Test_points.txt', test_points)
#
# Cheby_vals = []
# real_vals = []
# for i in range(n_test_points):
#     real_vals.append(expensive_func(*test_points[i]))
#     Cheby_vals.append(interp[test_points[i]])
#
# real_vals = np.array(real_vals)
# Cheby_vals = np.array(Cheby_vals)
#
# np.savetxt('Test_points_real_values.txt', real_vals)
# np.savetxt('Test_points_Cheby_values.txt', Cheby_vals)
#
# abs_errors = np.abs(real_vals - Cheby_vals)
# rel_errors = abs_errors / np.abs(real_vals)
#
# np.savetxt('Test_points_abs_errors.txt', abs_errors)
# np.savetxt('Test_points_rel_errors.txt', rel_errors)

abs_errors = np.loadtxt('Test_points_abs_errors.txt')
rel_errors = np.loadtxt('Test_points_rel_errors.txt')

print('\n+----+-----+----+----+----+----+----+-----+----+')

print('Minimum absolute error: ', min(abs_errors))
print('Maximum absolute error: ', max(abs_errors))
print('Median absolute error', np.median(abs_errors))
print('Mean absolute error', np.mean(abs_errors))
print('\nMinimum relative error: ', min(rel_errors))
print('Maximum relative error: ', max(rel_errors))
print('Median relative error', np.median(rel_errors))
print('Mean relative error', np.mean(rel_errors))

plt.hist(np.log10(rel_errors), density=False)
plt.title(f"ClassicCheb Order {order} distribution of relative errors for test function")
plt.xlabel('Base 10 logarithm of relative error')
plt.ylabel('Frequency')
plt.savefig(os.path.join(base_path, f'ClassicCheb_test_function_rel_errors_histogram.pdf'))
plt.show()

