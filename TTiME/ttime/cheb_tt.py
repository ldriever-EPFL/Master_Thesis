import os
import warnings
import numba
import numpy as np
from os.path import join
from copy import deepcopy
from .ad_rttc import AdRTTC
from .f_rttc_helper import inner_product
from .tt import TT


class ChebTT:
    """
    Implementation of the low-rank approximation Chebyshev interpolation method proposed in Glau 2020 {doi.org/10.1137/19M1244172}
    Will approximate a function using a tensorized Chebyshev interpolation, using only a limited number of function evaluations
    """

    def __init__(self, func, intervals, orders, max_r, max_func_evals, Omega_block_size, Gamma_size, storage_directory, **AdRTTC_kwargs):
        """
        :param func:            Callable that executes the to-be-interpolated function for a tuple of input values
        :param intervals        Intervals across which the Chebyshev interpolation shall be defined. Array of format dx2, with each row indicating one pair of minimum and maximum values for the parameter along the correspinding dimension
        :param orders           Chebyshev interpolation orders. Int (if the order is the same along all dimensions) or iterable of ints of length d
        :param max_r            Maximum rank to which the AdRTTC algorithm will go
        :Omega_block_size       Number of data points (i.e., evaluations of func) which are added to the training set with each batch
        :Gamma_size             Number of entries in the validation set
        :storage_directory      Path to an empty directory, where the results of the algorithm as well as the computed data will be stored
        :AdRTTC_kwargs          Dictionary with parameter specifications that are passed on to the AdRTTC algorithm. see ad_rttc.py for more information about the possible inputs
        """

        # Handle the inputs
        self.func = func
        self.intervals = np.array(intervals)
        self.d = len(self.intervals)
        self.orders = np.array(orders)
        if np.size(self.orders) == 1:
            # In case the interpolation order is the same along all dimensions
            self.orders = np.ones(len(self.intervals), dtype=int) * self.orders
        elif len(self.orders) != len(self.intervals):
            raise Exception("Orders must be a single integer or an iterable of length equal to that of intervals")

        if min(self.orders) < max_r - 1:
            raise Exception('The maximum rank is limited to the smallest number of interpolation points across all dimensions')

        self.n = self.orders + 1
        self.n_cumprod = np.cumprod(np.concatenate(([1], self.n[: -1])))
        self.max_r = max_r
        self.max_func_evals = max_func_evals
        self.Omega_block_size = Omega_block_size
        self.Gamma_size = Gamma_size
        self.storage_directory = storage_directory
        if not os.path.isdir(self.storage_directory) or len(os.listdir(self.storage_directory)) != 0:
            # Prevents unintentional overwriting of data, or unexpected errors later on after conducting expensive computations
            raise Exception("Storage directory must be an existing empty directory")

        np.savetxt(join(self.storage_directory, 'Intervals.txt'), self.intervals)

        if 'starter_cores' in AdRTTC_kwargs.keys():
            AdRTTC_kwargs.pop('starter_cores')

        self.converged = False

        # Create the points of the Chebyshev interval along each seperate dimension
        self.Cheb_points = []
        for i in range(self.d):
            self.Cheb_points.append(np.cos(np.pi * np.arange(self.n[i]) / self.orders[i]))

        self.Cheb_points = np.array(self.Cheb_points)

        # Create the affine transforms for mapping the intervals to the basic [-1, 1] interval used for Chebyshev interpolations
        self.a_transforms_to_Cheb = []
        for i in range(self.d):
            self.a_transforms_to_Cheb.append(lambda x, low=intervals[i][0], high=intervals[i][1]: (x - low) / (high - low) * 2 - 1)

        # Create the affine transforms for mapping the basic [-1, 1] Chebyshev interval to the actual parameter intervals
        self.a_transforms_from_Cheb = []
        for i in range(self.d):
            self.a_transforms_from_Cheb.append(lambda x, low=intervals[i][0], high=intervals[i][1]: (x + 1) / 2 * (high - low) + low)

        # Create the indices of testing and training points. This is cheap so we directly do it for all max_func_eval points, even if we won't use them all
        enumerated_ids = np.random.default_rng().choice(np.prod(self.n), size=self.max_func_evals, replace=False)
        idx_tuples = self.__idx_val_to_tuple(enumerated_ids).tolist()

        # Create the testing set and save it
        self.Gamma_indices = idx_tuples[:self.Gamma_size]
        og_Gamma_indices = deepcopy(self.Gamma_indices)
        np.savetxt(join(self.storage_directory, 'Gamma_indices.txt'), self.Gamma_indices)

        self.Gamma_vec = []
        self.Gamma_skipped_points = []
        for idcs in og_Gamma_indices:
            inpts = []
            for j in range(self.d):
                inpts.append(self.a_transforms_from_Cheb[j](self.Cheb_points[j][idcs[j]]))

            new_val = self.func(*inpts)

            if new_val is None:
                # If the evaluation of the expensive function failed we want to handle the failure but ensure that the program can keep on running
                # store the inputs that caused the failure
                self.Gamma_skipped_points.append(inpts)

                # Remove the index from the Gamma indices
                self.Gamma_indices.remove(idcs)
            else:
                self.Gamma_vec.append(new_val)

        self.Gamma_indices = np.array(self.Gamma_indices)
        self.Gamma_vec = np.array(self.Gamma_vec)
        np.savetxt(join(self.storage_directory, 'Gamma_vec.txt'), self.Gamma_vec)
        np.savetxt(join(self.storage_directory, 'Gamma_skipped_points.txt'), self.Gamma_skipped_points)

        # Run the AdRTTC algorithm for increasingly large training sets
        # Implements strategy 2 of algorithm 2 from Glau 2020 {doi.org/10.1137/19M1244172}
        self.N_completed_func_evals = self.Gamma_size
        self.Omega_vec = []
        self.Omega_idcs = []
        self.Omega_skipped_points = []
        count = 0
        self.P = None
        os.makedirs(join(self.storage_directory, 'Iterations'))

        while self.N_completed_func_evals + self.Omega_block_size <= self.max_func_evals and not self.converged:
            # Step 1: Create the training data, update Omega_vec, and save the result
            new_Omega_idcs = idx_tuples[self.Gamma_size + count * self.Omega_block_size : self.Gamma_size + (count + 1) * self.Omega_block_size]
            self.Omega_idcs += new_Omega_idcs
            for idcs in new_Omega_idcs:
                inpts = []
                for j in range(self.d):
                    inpts.append(self.a_transforms_from_Cheb[j](self.Cheb_points[j][idcs[j]]))

                new_val = self.func(*inpts)

                if new_val is None:
                    # If the evaluation of the expensive function failed we want to handle the failure but ensure that the program can keep on running
                    self.Omega_skipped_points.append(inpts)

                    # Remove the index from the Omega indices
                    self.Omega_idcs.remove(idcs)
                else:
                    self.Omega_vec.append(new_val)

            np.savetxt(join(self.storage_directory, 'Omega_indices.txt'), self.Omega_idcs)
            np.savetxt(join(self.storage_directory, 'Omega_vec.txt'), self.Omega_vec)
            np.savetxt(join(self.storage_directory, 'Omega_skipped_points.txt'), self.Omega_skipped_points)

            # Step 2: run the AdRTTC algorithm and store the results
            if self.P is None:
                opti_obj = AdRTTC(self.n, self.max_r, np.array(self.Omega_vec), np.array(self.Omega_idcs), self.Gamma_vec, self.Gamma_indices, **AdRTTC_kwargs)
            else:
                self.P.truncate_rank(np.ones(self.d - 1, dtype=int))
                opti_obj = AdRTTC(self.n, self.max_r, np.array(self.Omega_vec), np.array(self.Omega_idcs), self.Gamma_vec, self.Gamma_indices, starter_cores=self.P.cores, **AdRTTC_kwargs)

            self.P = opti_obj.X
            self.converged = opti_obj.converged

            os.makedirs(join(self.storage_directory, 'Iterations', f'Iteration{count}'))
            for i, core in enumerate(self.P.cores):
                np.savetxt(join(self.storage_directory, 'Iterations', f'Iteration{count}', f'coreL_{i}.txt'), core.L)

            np.savetxt(join(self.storage_directory, 'Iterations', f'Iteration{count}', f'testing_errors.txt'), opti_obj.testing_error_storage)
            np.savetxt(join(self.storage_directory, 'Iterations', f'Iteration{count}', f'training_errors.txt'), opti_obj.training_error_storage)
            np.savetxt(join(self.storage_directory, 'Iterations', f'Iteration{count}', f'rank_tuples.txt'), opti_obj.r_storage)

            # Step 3: update count and the number of completed function evaluations
            count += 1
            self.N_completed_func_evals += self.Omega_block_size

        # Save the optimization outcome
        os.makedirs(join(self.storage_directory, 'Final_cores'))
        for i, core in enumerate(self.P.cores):
            np.savetxt(join(self.storage_directory, 'Final_cores', f'coreL_{i}.txt'), core.L)

        with open(join(self.storage_directory, 'final_stats.txt'), 'w') as file:
            file.write(f'Number of iterations: {count}\n')
            file.write(f'Converged: {self.converged}\n')
            file.write(f'Number of conducted function evaluations: {self.N_completed_func_evals}\n')
            if hasattr(opti_obj, 'final_training_error'):
                file.write(f'Final training error: {opti_obj.final_training_error[-1]}\n')
                file.write(f'Final testing error: {opti_obj.final_testing_error[-1]}\n')
            elif self.converged:
                file.write(f'Final training error: {opti_obj.training_error_storage[-1]}\n')
                file.write(f'Final testing error: {opti_obj.testing_error_storage[-1]}\n')

        # Warn in case the algorithm did not convergence
        if not self.converged:
            warnings.warn("Algorithm did not convergence! Consider increasing the maximum rank or number of function evaluations")

        # Compute the C tensor
        self.__compute_C()

    def __idx_val_to_tuple(self, val):
        """
        The inverse operation of enumerating all entries in the tensor
        """
        idx_tuple_array = np.zeros((np.size(val), self.d), dtype=int)
        for i in range(self.d - 1, -1, -1):
            idx_tuple_array[:, i] = val // self.n_cumprod[i]
            val = val % self.n_cumprod[i]

        return idx_tuple_array

    def __compute_C(self):
        """
        Compute the tensor of Chebyshev coefficients in TT format using algorithm 3 of Glau 2020 {doi.org/10.1137/19M1244172}
        """
        C_TT = TT(self.P.cores)
        self.C = []
        for i in range(self.d):
            F = 2 / self.orders[i] * np.cos(np.pi * np.arange(self.n[i]) * np.arange(self.n[i]).reshape(-1, 1) / self.orders[i])
            F[0] /= 2
            F[-1] /= 2
            F[:, 0] /= 2
            F[:, -1] /= 2

            C_TT.mode_mu_multiply(F, i + 1)
            self.C.append(np.ascontiguousarray(np.array(C_TT.cores[i])))  # we can directly store the modified core, as it will not be further modified in future

    def __get_Cheb_basis_TT(self, point):
        """
        Computes the TT describing the tensorized Chebyshev polynomials for the input point
        """
        cores = []
        for i in range(self.d):
            cores.append(np.cos(np.arange(self.n[i]) * np.arccos(point[i])).reshape(1, -1, 1))

        return cores

    def __get_derived_Cheb_basis_TT(self, point, axis):
        """
        Calculates the  Chebyshev basis TT necessary for computing the derivative along axis 'axis'
        """
        cores = []
        for i in range(self.d):
            if i == axis:
                # We need to take care to avoid division by zero if point[i] == +- 1
                if point[i] == 1:
                    cores.append((np.arange(self.n[i]) * np.arange(self.n[i])).reshape(1, -1, 1) * 1.)  # *1. To ensure that the output is a float
                elif point[i] == -1:
                    core_contents = np.arange(self.n[i]) * np.arange(self.n[i])
                    core_contents[np.arange(self.n[i]) % 2 == 0] *= -1  # The terms corrresponding to even n must be multiplied by -1
                    cores.append(core_contents.reshape(1, -1, 1) * 1.)  # *1. To ensure that the output is a float
                else:
                    cores.append(
                        (np.sin(np.arange(self.n[i]) * np.arccos(point[i])) * np.arange(self.n[i])).reshape(1, -1, 1) / np.sqrt(1 - point[i] ** 2))
            else:
                cores.append(np.cos(np.arange(self.n[i]) * np.arccos(point[i])).reshape(1, -1, 1))

        return cores

    def __get_Cheb_inputs(self, point):
        """
        Maps an input from the original intervals to the Chebyshev [-1, 1] intervals
        """
        point = np.ravel(point)
        Cheb_inputs = []
        for i, val in enumerate(point):
            if not self.intervals[i][0] <= val <= self.intervals[i][1]:
                raise Exception(f'Provided value is outside the interpolation interval along axis {i}')

            Cheb_inputs.append(self.a_transforms_to_Cheb[i](val))

        return Cheb_inputs

    def __getitem__(self, item):
        Cheb_inputs = self.__get_Cheb_inputs(item)
        Cheb_basis_cores = self.__get_Cheb_basis_TT(Cheb_inputs)

        return inner_product(numba.typed.List(self.C), numba.typed.List(Cheb_basis_cores))

    def get_derivative(self, point, axis=None):
        """
        Returns the derivative vector of the interpolated function at the specified point.
        If axis is not None, but an integer or iterable of integers in the range(d), only the derivative along said axes will be computed
        """

        if axis is None:
            axis = np.arange(self.d)
        else:
            axis = np.ravel(axis)

        if not np.issubdtype(axis.dtype, int) or not np.all((axis >= 0) & (axis < self.d)):
            raise Exception(f"axis must be an integer or iterable of integers in the range [0, {self.d}]")

        # Transform the input point to the [-1, 1] Chebyshev interval
        Cheb_inputs = self.__get_Cheb_inputs(point)

        # Compute the desired derivatives
        derivative = [] * len(axis)
        for ax in axis:
            Cheb_derivative_basis_cores = self.__get_derived_Cheb_basis_TT(Cheb_inputs, ax)
            derivative.append(inner_product(numba.typed.List(self.C), numba.typed.List(Cheb_derivative_basis_cores)))
            derivative[-1] *= 2 / (self.intervals[ax, 1] - self.intervals[ax, 0])  # Account for the derivative of the affine transform to the Chebyshev domain

        return np.array(derivative)
