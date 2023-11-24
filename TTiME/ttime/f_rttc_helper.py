import numba
import numpy as np

"""
Provides a numba-compatible and accelerated version of the functions found in rttc_helper.py
"""


@numba.jit(nopython=True, cache=True)
def __get_r_n_d_from_cores(X: [np.ndarray]) -> (np.ndarray, np.ndarray, int):
    """
    Gets rank tuple, shape, and dimension of the TT tensor based on its cores
    """
    r = [1]
    n = []
    for core in X:
        r.append(core.shape[-1])
        n.append(core.shape[1])

    return np.array(r), np.array(n), len(X)


@numba.jit(nopython=True, cache=True)
def __get_cores_item(cores: [np.ndarray], item: np.ndarray) -> float:
    """
    Equivalent to the __getitem__() method of the TT class
    """
    result = cores[0][:, item[0], :]
    for i in range(1, len(item)):
        result = result @ cores[i][:, item[i], :]

    return result[0, 0]


@numba.jit(nopython=True, cache=True)
def __compute_alpha(Omega_indices_train: np.ndarray, direction_TT: [np.ndarray], A_vec_train: np.ndarray, X: [np.ndarray]) -> float:
    """
    Approximates the ideal step size for the line search
    """
    numerator = 0
    denominator = 0
    for count, idx in enumerate(Omega_indices_train):
        direction_term = __get_cores_item(direction_TT, idx)
        numerator += direction_term * (A_vec_train[count] - __get_cores_item(X, idx))
        denominator += direction_term * direction_term
    return numerator / denominator


@numba.jit(nopython=True, cache=True)
def __compute_error(X: [np.ndarray], A_vec: np.ndarray, Omega_indices: np.ndarray, A_norm: float) -> float:
    """
    Computes the error on the test set A_vec_test, relative to the magnitude of said set
    """
    numerator = 0
    for count, idx in enumerate(Omega_indices):
        numerator += (__get_cores_item(X, idx) - A_vec[count]) ** 2

    return np.sqrt(numerator) / A_norm


@numba.jit(nopython=True, cache=True)
def __core_copy(X: [np.ndarray]) -> [np.ndarray]:
    """
    Numba does not allow copy.deepcopy, so this functions as an alternative
    """
    Y = []
    for core in X:
        Y.append(np.copy(core))
    return numba.typed.List(Y)


@numba.jit(nopython=True, cache=True)
def __right_unfold(core: np.ndarray) -> np.ndarray:
    """
    Numba is not compatible with passing the 'order' kwarg to reshape, so we use transposing to work around
    """
    return np.ascontiguousarray(np.transpose(core, (0, 2, 1))).reshape((core.shape[0], core.shape[1] * core.shape[2]))


@numba.jit(nopython=True, cache=True)
def __left_unfold(core: np.ndarray) -> np.ndarray:
    """
    Numba is not compatible with passing the 'order' kwarg to reshape, so we use transposing to work around
    """
    return np.ascontiguousarray(np.transpose(core, (1, 0, 2))).reshape((core.shape[0] * core.shape[1], core.shape[2]))


@numba.jit(nopython=True, cache=True)
def __refold_right(matrix: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Numba is not compatible with passing the 'order' kwarg to reshape, so we use transposing to work around
    """
    return np.ascontiguousarray(np.transpose(matrix.reshape(-1, shape[0] * shape[1]).reshape(shape[0], shape[2], shape[1]), (0, 2, 1)))


@numba.jit(nopython=True, cache=True)
def __refold_left(matrix: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Numba is not compatible with passing the 'order' kwarg to reshape, so we use transposing to work around
    """
    return np.ascontiguousarray(np.transpose(np.ascontiguousarray(matrix.reshape(-1, shape[0] * shape[2]).T).reshape(shape[0], shape[2], shape[1]), (0, 2, 1)))


@numba.jit(nopython=True, cache=True)
def __mu_orthogonalize(X: [np.ndarray], mu: int) -> [np.ndarray]:
    # First go from the left
    for i in range(0, mu - 1):
        Q, R = np.linalg.qr(__left_unfold(X[i]))
        X[i] = __refold_left(np.copy(Q), (X[i].shape[0], X[i].shape[1], Q.shape[1]))
        X[i + 1] = __refold_right(np.copy(R) @ __right_unfold(X[i + 1]), (Q.shape[1], X[i + 1].shape[1], X[i + 1].shape[2]))

    # Then from the right
    for i in range(len(X) - 1, mu - 1, -1):
        Q, R = np.linalg.qr(__right_unfold(X[i]).T)
        X[i] = __refold_right(np.ascontiguousarray(Q.T), (Q.shape[1], X[i].shape[1], X[i].shape[2]))
        X[i - 1] = __refold_left(__left_unfold(X[i - 1]) @ R.T, X[i - 1].shape)

    return X


@numba.jit(nopython=True, cache=True)
def __left_orthogonalize(X: [np.ndarray]) -> [np.ndarray]:
    return __mu_orthogonalize(X, len(X))


@numba.jit(nopython=True, cache=True)
def __right_orthogonalize(X: [np.ndarray]) -> [np.ndarray]:
    return __mu_orthogonalize(X, 1)


@numba.jit(nopython=True, cache=True)
def __compute_riemannian_grad(U: [np.ndarray], V: [np.ndarray], X: [np.ndarray], A_vec: np.ndarray, Omega_indices: np.ndarray):
    """
    Computes the Riemannian gradient based on the Euclidean one
    """
    # Get dimension of X
    d = len(U)

    # Initialize the C tensors using zeros
    C = []
    for mu in range(d):
        C.append(np.zeros(U[mu].shape))

    # Loop through the sampling point indices, and thus through the non-zero entries of Z
    for count, idx in enumerate(Omega_indices):

        # Pre-compute the left matrix products for this sampling point
        U_products = [U[0][:, idx[0], :]]
        for mu in range(1, d - 1):
            U_products.append(U_products[-1] @ U[mu][:, idx[mu], :])

        # Add information from this sampling point to the C tensors
        Z_val = __get_cores_item(X, idx) - A_vec[count]
        V_product = np.array([[1.]])
        for mu in range(d - 1, 0, -1):
            C[mu][:, idx[mu], :] += U_products[mu - 1].T * V_product.T * Z_val
            V_product = V[mu][:, idx[mu], :] @ V_product
        C[0][:, idx[0], :] += V_product.T * Z_val

    # Find the first order variations by projecting the C tensors (appart from the last one) onto the orthogonal compliment of the range of the associated U.L
    # as U is known to be left orthogonal for mu up to and including d-1, U.L is already an orthonormal matrix, so no explicit QR decomposition is necessary
    dU = []
    for mu in range(d - 1):
        C_local_L = __left_unfold(C[mu])
        U_local_L = __left_unfold(U[mu])
        dU.append(__refold_left(C_local_L - U_local_L @ (U_local_L.T @ C_local_L), C[mu].shape))  # Smart matrix product factorization allows lowering the cost by a factor n
    dU.append(C[-1])  # The last variation is simply the C tensor

    # Return the first order variations
    return dU


@numba.jit(nopython=True, cache=True)
def __tangent_space_vector_transport(U: [np.ndarray], V: [np.ndarray], W: [np.ndarray]) -> [np.ndarray]:
    """
    Transports a tangent tensor Y with TT cores W into the tangent space of the rank-r manifold at X, where X has left-orthogonal cores U and right-orthogonal cores V
    """
    # Get dimension of X
    d = len(U)

    # Pre-compute the Y^T_{geq mu+1}X_{geq mu+1} terms, here called YTX
    YTX = [np.array([[1.]])]
    for mu in range(d - 1, 0, -1):
        kron_prod = np.zeros((W[mu].shape[2] * W[mu].shape[1], V[mu].shape[0]))
        for i in range(W[mu].shape[2]):  # loop over r
            for j in range(W[mu].shape[1]):  # loop over n
                a = YTX[0][i] @ __right_unfold(V[mu]).T[j:: W[mu].shape[1]]
                kron_prod[i * W[mu].shape[1] + j] = a

        YTX = [__right_unfold(W[mu]) @ kron_prod] + YTX
    # Iterate upwards through mu and compute the first order variations dU
    # Iteratively compute the (I_{n_mu} kronecker X_{leq mu-1})^T Y_{leq mu} terms, here called IXTY, on the fly
    dU = []
    IXTY = __left_unfold(W[0])

    for mu in range(d):
        dU_L = IXTY @ YTX[mu]

        if mu < d - 1:
            # All but the last variation must be projected onto the orthogonal compliment of the range of the associated U.L
            # as U is known to be left orthogonal for mu up to and including d-1, U.L is already an orthonormal matrix, so no explicit QR decomposition is necessary
            dU_L = dU_L - __left_unfold(U[mu]) @ (__left_unfold(U[mu]).T @ dU_L)  # Smart matrix product factorization allows lowering the cost by a factor n

            # Also update IXTY
            C = __left_unfold(U[mu]).T @ IXTY
            W_unfold = __left_unfold(W[mu + 1])
            new_IXTY = np.zeros((W[mu + 1].shape[1] * C.shape[0], W[mu + 1].shape[2]))
            for i in range(W[mu + 1].shape[1]):
                new_IXTY[i * C.shape[0] : (i + 1) * C.shape[0]] = C @ W_unfold[i * C.shape[1] : (i + 1) * C.shape[1]]

            IXTY = new_IXTY

        if len(dU_L.shape) == 1:
            dU_L = dU_L.reshape(-1, 1)
        dU.append(__refold_left(dU_L, U[mu].shape))

    # Return the first order variations
    return dU


@numba.jit(nopython=True, cache=True)
def __vars_to_TT(U: [np.ndarray], V: [np.ndarray], dU: [np.ndarray]) -> [np.ndarray]:
    """
    Turns a list of first order variations in the tangent space of X into TT cores
    Implements formulation 3.11 from Steinlechner 2016 {doi.org/10.1137/15M1010506}
    """
    new_cores = [np.concatenate((dU[0], U[0]), axis=-1)]
    for i in range(1, len(U) - 1):
        top_row = np.concatenate((V[i], np.zeros((V[i].shape[0], V[i].shape[1], U[i].shape[-1]))), axis=-1)
        bottom_row = np.concatenate((dU[i], U[i]), axis=-1)
        new_cores.append(np.concatenate((top_row, bottom_row), axis=0))

    new_cores.append(np.concatenate((V[-1], dU[-1]), axis=0))

    return new_cores


@numba.jit(nopython=True, cache=True)
def __update_X(U: [np.ndarray], V: [np.ndarray], alpha: float, dU: [np.ndarray]) -> [np.ndarray]:
    """
    Gives iteration X^{i+1} based on the cores of X^{i}, the step size, and the direction variations
    """
    # Implements the corrected version of the formulation on Steinlechner 2016 {doi.org/10.1137/15M1010506} pg 11. The first term on the RHS should have dU1 instead of V1
    # First scale the variations by the step size and add the last left-orthogonal core to the last scaled variation
    for i in range(len(dU)):
        dU[i] *= alpha

    dU[-1] += U[-1]

    # Then simply use the vars_to_TT function
    return __vars_to_TT(U, V, dU)


@numba.jit(nopython=True, cache=True)
def inner_product(X: [np.ndarray], Y: [np.ndarray]) -> float:
    """
    Calculates the inner product of two tensors in the TT format.
    Based on the approach implemented in the reference Matlab implementation provided for Steinlechner 2016 {doi.org/10.1137/15M1010506}
    """

    res = np.array([[1.]])
    for i in range(len(X)):
        temp = __refold_right(res.T @ __right_unfold(X[i]), (res.shape[1], X[i].shape[1], X[i].shape[2]))
        res = __left_unfold(temp).T @ __left_unfold(Y[i])

    return float(np.ravel(res)[0])


@numba.jit(nopython=True, cache=True)
def __truncate_rank(X: [np.ndarray], new_r: np.ndarray) -> [np.ndarray]:
    """
    Reduces the rank of X to that specified by new_r
    """
    # If the TT is not yet so, we must left-orthogonalize it
    X = __left_orthogonalize(X)

    # Then truncate the ranks, going from right to left
    for mu in range(len(X) - 1, 0, -1):
        X = __truncate_one_rank(X, new_r[mu - 1], mu)

    return X


@numba.jit(nopython=True, cache=True)
def __truncate_one_rank(X: [np.ndarray], new_r: int, mu: int) -> [np.ndarray]:
    """
    Truncate the mu^th rank (index mu in the rank array, located between cores with indices mu-1 and mu).
    It is assumed that the TT is known to be mu + 1 orthogonal
    """
    # Perform the SVD
    Q, S, VT = np.linalg.svd(__right_unfold(X[mu]))

    # Keep only the first 'new_r' singular values
    Q = Q[:, :new_r]
    S = np.diag(S[:new_r])
    VT = VT[:new_r]

    # Now adjust cores with indices mu and mu - 1
    X[mu] = __refold_right(np.ascontiguousarray(VT), (new_r, X[mu].shape[1], X[mu].shape[2]))
    X[mu - 1] = __refold_left(__left_unfold(X[mu - 1]) @ Q @ S, (X[mu - 1].shape[0], X[mu - 1].shape[1], new_r))

    return X


@numba.jit(nopython=True, cache=True)
def run_fast_rttc_algorithm(X: [np.ndarray], A_vec_train: np.ndarray, Omega_indices_train: np.ndarray, A_vec_test: np.ndarray, Omega_indices_test: np.ndarray, rel_error_threshold: float, error_stagnation_threshold: float, max_iter: int, verbose: bool) -> ([np.ndarray], [float], [float], str):
    """
    Combines all numba-accelerated functions into a fast version of the RTTC algorithm
    Returns cores, converged_bool, training_error, test_error, exit_message
    """
    # Set up storage
    converged = False
    training_error = []
    testing_error = []

    # Set up auxilliary variables
    A_train_norm = np.linalg.norm(A_vec_train)
    A_test_norm = np.linalg.norm(A_vec_test)
    r, n, d = __get_r_n_d_from_cores(X)
    
    # Define arbitrary initial states for parameters only defined in the second loop iteration (necessary for Numba)
    prev_grad_cores = X
    prev_direction_cores = X
    prev_grad_norm = 1.
    
    # Check if initial guess already satisifies the convergence conditions
    training_error.append(__compute_error(X, A_vec_train, Omega_indices_train, A_train_norm))
    testing_error.append(__compute_error(X, A_vec_test, Omega_indices_test, A_test_norm))

    if verbose:
        print('-----------------------------------------------')
        print('Initial training error: ', training_error[-1])
        print('Initial testing error: ', testing_error[-1])

    if training_error[-1] < rel_error_threshold and testing_error[-1] < rel_error_threshold:
        converged = True
        exit_message = "Initial guess X0 already fulfills the error tolerance for both training and testing error"
        return X, converged, training_error, testing_error, exit_message
    
    for iter_count in range(max_iter):
        if verbose:
            print('-----------------------------------------------')
            print('Iteration: ', iter_count)

        # Compute the Riemannian Gradient
        X = __left_orthogonalize(X)
        U = __core_copy(X)
        X = __right_orthogonalize(X)
        V = __core_copy(X)

        grad_vars = __compute_riemannian_grad(U, V, X, A_vec_train, Omega_indices_train)  # first order variations describing the Riemannian gradient

        # We either take a steepest descent step or use the conjugate gradient Scheme with Fletcher-Reeves update to find the new search direction
        grad_norm = 0.
        for dU in grad_vars:
            grad_norm += np.linalg.norm(dU.flatten()) ** 2

        grad_norm = np.sqrt(grad_norm)
        grad_TT = __vars_to_TT(U, V, grad_vars)

        if iter_count == 0:
            direction_vars = []
            for var in grad_vars:
                direction_vars.append(-var)
            if verbose:
                print("Steepest descent step")
        else:
            # We use the restart criterion implemented in the reference implementation of Steinlechner 2016 {doi.org/10.1137/15M1010506} (see https://www.epfl.ch/labs/anchp/index-html/software/ttemps/)
            __tangent_space_vector_transport(U, V, prev_grad_cores)
            prev_grad_TT_at_X = __vars_to_TT(U, V, __tangent_space_vector_transport(U, V, prev_grad_cores))

            theta = inner_product(grad_TT, prev_grad_TT_at_X) / grad_norm ** 2
            if verbose:
                print('Theta: ', theta)
            if abs(theta) >= 0.1:
                # The current and previous gradients are insufficiently aligned so we do a direction reset
                direction_vars = []
                for var in grad_vars:
                    direction_vars.append(-var)

                if verbose:
                    print("Steepest descent step")
            else:
                # We use the conjugate gradient scheme. Now that a previous direction exists, transport it to the tangent space at X
                prev_direction_vars_at_X = __tangent_space_vector_transport(U, V, prev_direction_cores)
                beta = (grad_norm / prev_grad_norm) ** 2
                direction_vars = []
                for i in range(d):
                    direction_vars.append(-grad_vars[i] + beta * prev_direction_vars_at_X[i])

                if verbose:
                    print("CG step")

        direction_TT = __vars_to_TT(U, V, direction_vars)

        # Find the approximately optimal step size
        alpha = __compute_alpha(Omega_indices_train, direction_TT, A_vec_train, X)

        # Do the step
        X = __update_X(U, V, alpha, direction_vars)

        # Retract back to the rank-r manifold
        X = __truncate_rank(X, r[1: -1])  # Ignore the leading and trailing 1 of the rank array

        X = numba.typed.List(X)

        # Check for convergence
        training_error.append(__compute_error(X, A_vec_train, Omega_indices_train, A_train_norm))
        testing_error.append(__compute_error(X, A_vec_test, Omega_indices_test, A_test_norm))

        if verbose:
            print('Training error: ', training_error[-1])
            print('Testing error: ', testing_error[-1])

        if training_error[-1] < rel_error_threshold and testing_error[-1] < rel_error_threshold:
            converged = True
            exit_message = f"Algorithm converged after {iter_count + 1} iterations"
            return X, converged, training_error, testing_error, exit_message

        # Check for stagnation
        training_stagnated = (training_error[-2] - training_error[-1]) / training_error[-2] < error_stagnation_threshold
        testing_stagnated = (testing_error[-2] - testing_error[-1]) / testing_error[-2] < error_stagnation_threshold

        if training_stagnated and testing_stagnated:
            exit_message = f"Both training and testing error stagnated after {iter_count + 1} iterations"
            return X, converged, training_error, testing_error, exit_message
        if training_stagnated:
            exit_message = f"Training error stagnated after {iter_count + 1} iterations"
            return X, converged, training_error, testing_error, exit_message

        if testing_stagnated:
            exit_message = f"Testing error stagnated after {iter_count + 1} iterations"
            return X, converged, training_error, testing_error, exit_message

        # Store values for the following iteration
        prev_direction_cores = numba.typed.List(direction_TT)
        prev_grad_cores = numba.typed.List(grad_TT)
        prev_grad_norm = grad_norm

    # If this part of the code is reached it means that max_iter has been reached
    exit_message = "The maximum number of iterations has been reached without convergence or stagnation"
    return X, converged, training_error, testing_error, exit_message
