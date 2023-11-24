import time
import numba
import numpy as np
import warnings
from copy import deepcopy
from .rttc_helper import compute_riemannian_grad, tangent_space_vector_transport, vars_to_TT, update_X, inner_product
from .f_rttc_helper import run_fast_rttc_algorithm
from .tt import TT
from tqdm import tqdm


class RTTC:
    """
    Implements the Riemannian CG for Tensor Train Completion (RTTC) algorithm laid out in Steinlechner 2016 {doi.org/10.1137/15M1010506}
    """
    def __init__(self, X0: TT, A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rel_error_threshold=1e-4, error_stagnation_threshold=1e-4, max_iter=1e5, verbose=False, fast=True):
        """
        The init function will automatically run the Riemannian CG to fit X to the sample points in A. The result of the fitting can be accessed using obj.X

        :param X0:                          Tensor in TT format (must have type TT). Initial guess for the CG algorithm. Its TT rank defines the manifold on which the fitting is done
        :param A_vec_train:                 Vector of data values at the sampling points of the training set
        :param Omega_indices_train:         The indices of the training points in the data tensor A. Formatted as array of size |Omega_train| x d. Order of points must agree with that in A_vec_train
        :param A_vec_test:                  Vector of data values at the sampling points of the testing set
        :param Omega_indices_test:          The indices of the testing points in the data tensor A. Formatted as array of size |Omega_test| x d. Order of points must agree with that in A_vec_test
        :param rel_error_threshold:         If the relative error falls below this value for both testing and training set, the algorithm is considered to have converged
        :param error_stagnation_threshold:  If the relative decrease in error between consecutive iterations is less than this value, the algorithm is deemed to have stagnated and is terminated
        :param max_iter:                    Maximum number of iterations, after which the algorithm terminates if there is still no convergence
        :param verbose:                     Bool. If True, the testing and training error are displayed alonside the iteration number at each step
        :param fast:                        Bool. If True, the numba-accelerated version of the code is used
        """
        # Store the input parameters
        self.X = X0
        self.r = self.X.r
        self.A_vec_train = A_vec_train
        self.Omega_indices_train = Omega_indices_train
        self.A_vec_test = A_vec_test
        self.Omega_indices_test = Omega_indices_test
        self.rel_error_threshold = rel_error_threshold
        self.error_stagnation_threshold = error_stagnation_threshold
        self.max_iter = int(max_iter)
        self.verbose = verbose

        # Check the size of Omega_train. It should be at least O(dnr^2), the dimension of the rank-r manifold (see Steinlechner 2016 {doi.org/10.1137/15M1010506} pg 6, 12)

        manifold_dim = - sum(self.X.r[1: -1] ** 2)
        for i in range(self.X.d):
            manifold_dim += self.X.r[i] * self.X.r[i + 1] * self.X.n[i]

        if len(A_vec_train) < manifold_dim:
            raise Exception(f"Training set should at least contain {manifold_dim} points")

        if fast:
            # Run the fast RTTC algorithm
            new_cores, self.converged, self.training_error, self.testing_error, self.exit_message = run_fast_rttc_algorithm(numba.typed.List(self.X.cores), A_vec_train, Omega_indices_train, A_vec_test, Omega_indices_test, rel_error_threshold, error_stagnation_threshold, int(max_iter), verbose)

            self.X = TT(new_cores)

        else:
            # Create the attributes used for tracking the CG progress
            self.converged = False
            self.exit_message = None
            self.training_error = []
            self.testing_error = []

            # Run the RTTC algorithm
            self.__run_rttc_algorithm()

        # Issue a warning if the algorithm did not converge
        if verbose and not self.converged:
            warnings.warn(f"The RTTC algorithm did not converge. Exit message: {self.exit_message}")

    def __run_rttc_algorithm(self):
        """
        Runs at most max_iter iterations of the RTTC algorithm, terminating if the error falls below the specified tolerance, or if progress stagnates
        Implements 'Algorithm 1' of Steinlechner 2016 {doi.org/10.1137/15M1010506}

        :return: None
        """
        # Check if initial guess already satisifies the convergence conditions
        self.training_error.append(self.__compute_training_error())
        self.testing_error.append(self.__compute_testing_error())

        if self.verbose:
            print('-----------------------------------------------')
            print('Initial training error: ', self.training_error[-1])
            print('Initial testing error: ', self.testing_error[-1])

        if self.training_error[-1] < self.rel_error_threshold and self.testing_error[-1] < self.rel_error_threshold:
            self.converged = True
            self.exit_message = "Initial guess X0 already fulfills the error tolerance for both training and testing error"
            return
        
        for iter_count in tqdm(range(self.max_iter), disable=self.verbose):
            if self.verbose:
                print('-----------------------------------------------')
                print('Iteration: ', iter_count)
                start = time.time()

            # Compute the Riemannian Gradient
            self.X.left_orthogonalize()
            U = deepcopy(self.X.cores)
            self.X.right_orthogonalize()
            V = deepcopy(self.X.cores)
            grad_vars = compute_riemannian_grad(U, V, self.X, self.A_vec_train, self.Omega_indices_train)  # first order variatinos describing the RIemannian gradient

            # We either take a steepest descent step or use the conjugate gradient Scheme with Fletcher-Reeves update to find the new search direction
            grad_norm = np.sqrt(sum(np.linalg.norm(dU) ** 2 for dU in grad_vars))
            grad_TT = vars_to_TT(U, V, grad_vars)

            if iter_count == 0:
                direction_vars = [-var for var in grad_vars]
                if self.verbose:
                    print("Steepest descent step")
            else:
                # We use the restart criterion implemented in the reference implementation of Steinlechner 2016 {doi.org/10.1137/15M1010506} (see https://www.epfl.ch/labs/anchp/index-html/software/ttemps/)
                prev_grad_TT_at_X = vars_to_TT(U, V, tangent_space_vector_transport(U, V, prev_grad_cores))

                theta = inner_product(grad_TT, prev_grad_TT_at_X) / grad_norm ** 2
                if theta >= 0.1:
                    # The current and previous gradients are insufficiently aligned so we do a direction reset
                    direction_vars = [-var for var in grad_vars]
                    if self.verbose:
                        print("Steepest descent step")
                else:
                    # We use the conjugate gradient scheme. Now that a previous direction exists, transport it to the tangent space at X
                    prev_direction_vars_at_X = tangent_space_vector_transport(U, V, prev_direction_cores)
                    # TODO discuss how this is the beta used in the ref implementation, but is not the original Fletcher Reeves discussed in the paper
                    beta = (grad_norm / prev_grad_norm) ** 2
                    direction_vars = [-grad_vars[i] + beta * prev_direction_vars_at_X[i] for i in range(self.X.d)]
                    if self.verbose:
                        print("CG step")

            direction_TT = vars_to_TT(U, V, direction_vars)

            # Find the approximately optimal step size
            numerator = 0
            denominator = 0
            for count, idx in enumerate(self.Omega_indices_train):
                direction_term = direction_TT[idx]
                numerator += direction_term * (self.A_vec_train[count] - self.X[idx])
                denominator += direction_term * direction_term
            alpha = numerator / denominator

            # Do the step
            self.X = update_X(U, V, alpha, direction_vars)

            # Retract back to the rank-r manifold
            self.X.truncate_rank(self.r[1: -1])  # Ignore the leading and trailing 1 of the rank array

            # Check for convergence
            self.training_error.append(self.__compute_training_error())
            self.testing_error.append(self.__compute_testing_error())

            if self.verbose:
                print('Training error: ', self.training_error[-1])
                print('Testing error: ', self.testing_error[-1])

            if self.training_error[-1] < self.rel_error_threshold and self.testing_error[-1] < self.rel_error_threshold:
                self.converged = True
                self.exit_message = f"Algorithm converged after {iter_count + 1} iterations"
                return

            # Check for stagnation
            # TODO implement different stagnatino criterion, something that allows bounded increases (either a limited number of successive increases or an upper error limit based on the best error achieved so far)
            training_stagnated = (self.training_error[-2] - self.training_error[-1]) / self.training_error[-2] < self.error_stagnation_threshold
            testing_stagnated = (self.testing_error[-2] - self.testing_error[-1]) / self.testing_error[-2] < self.error_stagnation_threshold

            if training_stagnated and testing_stagnated:
                self.exit_message = f"Both training and testing error stagnated after {iter_count + 1} iterations"
                return
            if training_stagnated:
                self.exit_message = f"Training error stagnated after {iter_count + 1} iterations"
                return

            if testing_stagnated:
                self.exit_message = f"Testing error stagnated after {iter_count + 1} iterations"
                return

            # Store values for the following iteration
            prev_direction_cores = direction_TT.cores
            prev_grad_cores = grad_TT.cores
            prev_grad_norm = grad_norm

            if self.verbose:
                print('Iteration duration: ', time.time() - start)

        # If this part of the code is reached it means that max_iter has been reached
        self.exit_message = "The maximum number of iterations has been reached without convergence or stagnation"

    def __compute_training_error(self):
        """
        Computes the error on the training set A_vec_train, relative to the magnitude of said set
        """
        numerator = 0
        for count, idx in enumerate(self.Omega_indices_train):
            numerator += (self.X[idx] - self.A_vec_train[count]) ** 2

        return float(np.sqrt(numerator) / np.linalg.norm(self.A_vec_train))

    def __compute_testing_error(self):
        """
        Computes the error on the test set A_vec_test, relative to the magnitude of said set
        """
        numerator = 0
        for count, idx in enumerate(self.Omega_indices_test):
            numerator += (self.X[idx] - self.A_vec_test[count]) ** 2

        return float(np.sqrt(numerator) / np.linalg.norm(self.A_vec_test))
