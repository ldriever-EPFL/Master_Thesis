import os
import sys
import numpy as np
from func_definition import expensive_func
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from ttime import ChebTT

"""
Empty the directory 'cheb_outputs' before running this file

If the file runs without raising an Exception, the test is passed
"""

np.random.seed(77)

# Define the inputs
intervals = np.array([[0, 1],
                     [2, 3],
                     [-1, 3],
                     [1, 4],
                     [0.01, 0.03]])

orders = 10
max_r = 10
max_func_evals = 5300
Omega_block_size = 500
Gamma_size = 300
storage_directory = os.path.join(base_path, 'cheb_outputs')

AdRTTC_kwargs = {'rho': 0.5,
                    'error_stagnation_threshold_coarse': 1e-2,
                    'error_stagnation_threshold_fine': 0,
                    'final_error_stagnation_threshold': -1e-4,
                    'best_r_err_tol': 1e-3,
                    'seed': 99,
                    'rel_error_threshold': 1e-4,
                    'max_iter_coarse': 20,
                    'max_iter_fine': 40,
                    'r_max_coarse': 3,
                    'final_max_iter': 100,
                    'verbose': True,
                    'super_verbose': False,
                    'final_verbose': False
                    }

# Run the training
interp = ChebTT(expensive_func, intervals, orders, max_r, max_func_evals, Omega_block_size, Gamma_size, storage_directory, **AdRTTC_kwargs)

# Test the interpolation for 1000 random points in the specified interval
n_test_points = 1000
test_points = np.random.uniform(intervals[:, 0], intervals[:, 1], size=(n_test_points, len(intervals)))

np.savetxt('Test_points.txt', test_points)

Cheby_vals = []
real_vals = []
for i in range(n_test_points):
    real_vals.append(expensive_func(*test_points[i]))
    Cheby_vals.append(interp[test_points[i]])

real_vals = np.array(real_vals)
Cheby_vals = np.array(Cheby_vals)

np.savetxt('Test_points_real_values.txt', real_vals)
np.savetxt('Test_points_Cheby_values.txt', Cheby_vals)

abs_errors = np.abs(real_vals - Cheby_vals)
rel_errors = abs_errors / real_vals

np.savetxt('Test_points_abs_errors.txt', abs_errors)
np.savetxt('Test_points_rel_errors.txt', rel_errors)

print('\n+----+-----+----+----+----+----+----+-----+----+')

print('Minimum absolute error: ', min(abs_errors))
print('Maximum absolute error: ', max(abs_errors))
print('Average absolute error', np.mean(abs_errors))
print('\nMinimum relative error: ', min(rel_errors))
print('Maximum relative error: ', max(rel_errors))
print('Average relative error', np.mean(rel_errors))
