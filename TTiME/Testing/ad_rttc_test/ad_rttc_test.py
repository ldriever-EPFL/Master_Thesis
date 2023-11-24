import os
import sys
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../../'))
from ttime import AdRTTC

"""
Tests the implementation of the AdRTTC class

If the file runs without raising an Exception, the test is passed
"""

base_path = os.path.dirname(os.path.abspath(__file__))
np.random.seed(99)

# Define the key parameters
d = 5
shape = np.ones(d, dtype=int) * 10
r_max = 10
rho = 1e-2

# As testing case we adapt the example shown in Glau 2020 {doi.org/10.1137/19M1244172} pg 10

# UNCOMMENT TO GENERATE DATA
# n_train = 1000
# n_test = 5e4
#
# idcs = set()
# while len(idcs) < n_test + n_train:
#     new_idx = tuple(np.random.randint(shape))
#     idcs.add(new_idx)
#
# idcs = np.array(list(idcs), dtype=int)
# np.savetxt('train_idcs.txt', idcs[:n_train])
# np.savetxt('test_idcs.txt', idcs[n_train:])
#
# A = np.exp(-np.linalg.norm(idcs / 20, axis=1))
# np.savetxt('A_train.txt', A[:n_train])
# np.savetxt('A_test.txt', A[n_train:])

idcs_train = np.loadtxt(os.path.join(base_path, "train_idcs.txt")).astype(np.int64)
idcs_test = np.loadtxt(os.path.join(base_path, "test_idcs.txt")).astype(np.int64)
A_train = np.loadtxt(os.path.join(base_path, "A_train.txt"))
A_test = np.loadtxt(os.path.join(base_path, "A_test.txt"))

# Run the AdRTTC algorithm
opti_TT = AdRTTC(shape, r_max, A_train, idcs_train, A_test, idcs_test, rho=rho, error_stagnation_threshold_coarse=1e-2, error_stagnation_threshold_fine=0, max_iter_fine=80, final_error_stagnation_threshold=-1e-4, best_r_err_tol=0, seed=99)
