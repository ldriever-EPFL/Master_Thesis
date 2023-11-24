import os
import sys
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../../'))
from ttime import RTTC, TT

"""
Tests the implementation of the RTTC class (and thus the implementation of the Riemannian CG for Tensor Train Completion)
through comparison with the MATLAB code provided by Steinlechner in connection with his 2016 paper {doi.org/10.1137/15M1010506}

If the file runs without raising an Exception, the test is passed
"""

base_path = os.path.dirname(os.path.abspath(__file__))

# Define the key parameters
d = 10
n = np.ones(d, dtype=np.int64) * 20
r = np.array([1, 3, 4, 2, 6, 4, 4, 3, 5, 3, 1])

n_train = d * n[0] * r[1] ** 2 * 10
n_test = n_train

# Set the seed for the random number generator
np.random.seed(99)

# Create a random low-rank order d tensor using the TT class
theoretical_cores = []
for i in range(d):
    theoretical_cores.append(np.random.random((r[i + 1], n[i], r[i])).T)  # Transpose gives same cores as for MATLAB

A = TT(theoretical_cores)

# Random starting guess (same as in MATLAB)
theoretical_cores = []
for i in range(d):
    theoretical_cores.append(np.random.random((r[i + 1], n[i], r[i])).T)  # Transpose gives same cores as for MATLAB

X0 = TT(theoretical_cores)
X0.left_orthogonalize()

# Load the indices and data also used in the MATLAB case
idcs_train = np.loadtxt(os.path.join(base_path, "train_idcs.txt"), delimiter=',', dtype=np.int64) - 1
idcs_test = np.loadtxt(os.path.join(base_path, "test_idcs.txt"), delimiter=',', dtype=np.int64) - 1

A_train = np.loadtxt(os.path.join(base_path, "A_train.txt"))
A_test = np.loadtxt(os.path.join(base_path, "A_test.txt"))

# Run the optimization
opti_TT = RTTC(X0, A_train, idcs_train, A_test, idcs_test, max_iter=40, rel_error_threshold=1e-14, error_stagnation_threshold=0, verbose=True, fast=True)
opti_TT.X.left_orthogonalize()

# Save the results
np.savetxt(os.path.join(base_path, "rttc_output_files/training_error.txt"), opti_TT.training_error)
np.savetxt(os.path.join(base_path, "rttc_output_files/testing_error.txt"), opti_TT.testing_error)

for i in range(len(n)):
    np.savetxt(os.path.join(base_path, f"rttc_output_files/core{i}.txt"), opti_TT.X.cores[i].L)

# Check that the results agree with the MATLAB implementation
np.testing.assert_array_almost_equal(np.loadtxt(os.path.join(base_path, 'matlab_files/rttc_matlab_testing_error.txt')), opti_TT.testing_error[1:])
np.testing.assert_array_almost_equal(np.loadtxt(os.path.join(base_path, 'matlab_files/rttc_matlab_training_error.txt')), opti_TT.training_error[1:])
for i in range(d):
    np.testing.assert_array_almost_equal(np.loadtxt(os.path.join(base_path, f'matlab_files/rttc_matlab_core_{i}.txt'), delimiter=','), opti_TT.X.cores[i].reshape(-1, order='F'))
