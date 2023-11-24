import warnings
import numba
import numpy as np
from copy import deepcopy
from .f_rttc_helper import run_fast_rttc_algorithm
from .tt import TT


class AdRTTC:
    """
    Implements an adaptation of the Adaptive RTTC algorithm laid out in Glau 2020 {doi.org/10.1137/19M1244172} Algorithm 1
    """

    def __init__(self, shape, r_max, A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rho=0, rel_error_threshold=1e-10, error_stagnation_threshold_coarse=1e-2, error_stagnation_threshold_fine=1e-4, max_iter_coarse=20, max_iter_fine=80, r_max_coarse=3, best_r_err_tol=0, final_error_stagnation_threshold=0, final_max_iter=100, starter_cores=None, seed=None, verbose=True, super_verbose=False, final_verbose=False):
        """
        :param shape:                               Iterable of ints. shape of the to-be-replicated Tensor
        :param r_max:                               int. maximally accepted rank
        :param A_vec_train:                         Vector of data values at the sampling points of the training set
        :param Omega_indices_train:                 The indices of the training points in the data tensor A. Formatted as array of size |Omega_train| x d. Order of points must agree with that in A_vec_train
        :param A_vec_test:                          Vector of data values at the sampling points of the testing set
        :param error_stagnation_threshold_coarse:   Large (in relative terms) for the coarse runs. If the relative decrease in error between consecutive iterations is less than this value, the algorithm is deemed to have stagnated and is terminated
        :param error_stagnation_threshold_fine:     As above, but for the more extensive runs. Usually small
        :param Omega_indices_test:                  The indices of the testing points in the data tensor A. Formatted as array of size |Omega_test| x d. Order of points must agree with that in A_vec_test
        :param rho:                                 float >= 0. Determines maximally allowed error increase for a rank to be accepted
        :param rel_error_threshold:                 If the relative error falls below this value for both testing and training set, the algorithm is considered to have converged
        :param max_iter_coarse:                     int. Small (in relative terms) for coarse runs Maximum number of CG iterations that are run before the algorithm is terminated without converging
        :param max_iter_fine:                       As above but for the more extensive runs. Usually large
        :param r_max_coarse:                        Rank increases up to and including this rank are deemed unlikely to be the solution as the ranks are small. Thus we use the coarse parameters (and hence cheaper simulations) before moving on to higher r
        :param best_r_err_tol:                      The best rank will be defined as the lowest rank tuple whose testing error is within best_r_err_tol relative to the lowest attained testing error across all r. Allows lowering complexity without losing accuracy. 0 menas the best value will be used. 1e-3 is recommended otherwise
        :param final_error_stagnation_threshold:    If not None, the best rank tuple is used for a final training run with this stagnation threshold
        :param final_max_iter:                      If final_error_stagnetion_threshold is not None, this max_iter is used for a final training run for the best r
        :param starter_cores:                       If not None, these are used to initialize X instead of the usual random rank 1 cores
        :param seed:                                If not None, used as the seed for the rng used to create the initial guess
        :param verbose:                             If True, updates are printed about the progress of the adaptive RTTC search
        :param super_verbose:                       If True, verbose is set to True for each individual run of the RTTC algorithm
        :param final_verbose:                       If True, verbose is set to True for the final training run in case that final_error_stagnetion_threshold is not None
        """

        if seed is not None:
            np.random.seed(seed)

        d = len(shape)
        if starter_cores is None:
            r = np.ones(d + 1, dtype=int)
        else:
            r = [1]
            for core in starter_cores:
                r.append(core.shape[-1])
            r = np.array(r)

        self.r_storage = [r]
        self.testing_error_storage = []
        self.training_error_storage = []
        self.converged = False

        # Set up the algorithm by creating a random initial guess and running one coarse iteration of RTTC
        if starter_cores is None:
            starter_cores = []
            for i in range(d):
                starter_cores.append(np.random.random((r[i], shape[i], r[i + 1])))

        new_cores, local_converged, local_training_error, local_testing_error, _ = run_fast_rttc_algorithm(numba.typed.List(starter_cores), A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rel_error_threshold, error_stagnation_threshold_coarse, max_iter_coarse, super_verbose)
        self.X = TT(new_cores)
        self.training_error_storage.append(local_training_error[-1])
        self.testing_error_storage.append(local_testing_error[-1])

        if verbose:
            print('+----+-----+----+----+----+----+----+-----+----+')
            print(f'Current rank tuple: {r}')
            print(f'Current training error: {self.training_error_storage[-1]}')
            print(f'Current testing error: {self.testing_error_storage[-1]}')
            print('Step trivially accepted')

        if local_converged:
            self.converged = True
            if verbose:
                print('+----+-----+----+----+----+----+----+-----+----+')
                print(f'Algorithm converged for rank tuple: {r}')
            return

        fruitless_attempts = 0
        mu = 1
        best_testing_error = 1e10
        best_X = self.X

        # Iterate until there is no more progress or all ranks have attained the maximum value
        while fruitless_attempts < d - 1 and np.any(r < r_max):
            # Increase the mu^th rank of X and run RTTC. If the new rank is less than equal r_max_coarse, use the coarse parameters. If the new r would exceed r_max, skip this iteration
            if r[mu] == r_max:
                fruitless_attempts += 1
            else:
                X_new = deepcopy(self.X)
                X_new.random_vec_rank_increase(mu)

                if r[mu] < r_max_coarse:
                    new_cores, local_converged, local_training_error, local_testing_error, msg = run_fast_rttc_algorithm(numba.typed.List(X_new.cores), A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rel_error_threshold, error_stagnation_threshold_coarse, max_iter_coarse, super_verbose)
                else:
                    new_cores, local_converged, local_training_error, local_testing_error, msg = run_fast_rttc_algorithm(numba.typed.List(X_new.cores), A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rel_error_threshold, error_stagnation_threshold_fine, max_iter_fine, super_verbose)

                if verbose:
                    print('+----+-----+----+----+----+----+----+-----+----+')
                    print(f'Current rank tuple: {X_new.r}')
                    print(f'RTTC terminated with exit message: {msg}')
                    print(f'Current training error: {local_training_error[-1]}')
                    print(f'Current testing error: {local_testing_error[-1]}')

                if local_converged:
                    self.r_storage.append(X_new.r)
                    self.training_error_storage.append(local_training_error[-1])
                    self.testing_error_storage.append(local_testing_error[-1])
                    self.converged = True
                    self.X = TT(new_cores)
                    r[mu] += 1
                    if verbose:
                        print('+----+-----+----+----+----+----+----+-----+----+')
                        print(f'Algorithm converged for rank tuple: {r}')
                    return
                elif local_testing_error[-1] - self.testing_error_storage[-1] < rho * self.testing_error_storage[-1]:
                    self.r_storage.append(X_new.r)
                    self.training_error_storage.append(local_training_error[-1])
                    self.testing_error_storage.append(local_testing_error[-1])

                    fruitless_attempts = 0
                    self.X = TT(new_cores)
                    r[mu] += 1
                    if verbose:
                        print('Step accepted')

                    if (best_testing_error - local_testing_error[-1]) / best_testing_error > best_r_err_tol:
                        best_testing_error = local_testing_error[-1]
                        best_X = deepcopy(self.X)

                else:
                    fruitless_attempts += 1
                    if verbose:
                        print('Step rejected')

            mu = np.mod(mu, d - 1) + 1

        self.testing_error_storage = np.array(self.testing_error_storage)
        self.X = best_X

        if verbose:
            print('+----+-----+----+----+----+----+----+-----+----+')
            print(f'Best rank tuple: {self.X.r}')
            print(f'...with associated testing error {best_testing_error}')

        if final_error_stagnation_threshold is not None:
            new_cores, local_converged, local_training_error, local_testing_error, msg = run_fast_rttc_algorithm(numba.typed.List(self.X.cores), A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rel_error_threshold, final_error_stagnation_threshold, final_max_iter, final_verbose)
            self.X = TT(new_cores)
            self.final_training_error = local_training_error
            self.final_testing_error = local_testing_error
            self.converged = local_converged

            if verbose:
                print('+----+-----+----+----+----+----+----+-----+----+')
                print(f'Final training on best rank completed with message: {msg}')
                print(f'Final training error: {local_training_error[-1]}')
                print(f'Final testing error: {local_testing_error[-1]}')

        if super_verbose and not self.converged:
            warnings.warn(f"The AdRTTC algorithm did not converge.")
