import os
import numpy as np
from ttime import preChebTT

"""
Test that the results of running ChebTT in main.py can be loaded into an instance of the preChebTT class and that the result is the same
Make sure to run cheb_test.py first
"""

# Create the preChebTT instance
intervals = np.array([[0, 1],
                     [2, 3],
                     [-1, 3],
                     [1, 4],
                     [0.01, 0.03]])

core_storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cheb_outputs', 'Final_cores')
interval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cheb_outputs', 'Intervals.txt')

interp = preChebTT(core_storage_path, interval_path)

# Load the testing points and results on the original ChebTT object
test_points = np.loadtxt('Test_points.txt')
ChebTT_vals = np.loadtxt('Test_points_Cheby_values.txt')

for i, point in enumerate(test_points):
    new_val = interp[point]

    if np.round(new_val, 15) != np.round(ChebTT_vals[i], 15):
        raise Exception(f"ChebTT and preChebTT yield different values for test point {i}")

print('All tests passed')



