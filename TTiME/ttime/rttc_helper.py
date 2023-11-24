import numpy as np
import scipy.sparse as spsp
from.tensor import Tensor
from .tt_core import TTCore
from .tt import TT

"""
Provides helper functions for the RTTC class. For the numba-accelerated version of the functions, see f_rttc_helper.py
"""


def compute_riemannian_grad(U, V, X, A_vec, Omega_indices):
    """
    Finds the Riemannian gradiant by projecting the Euclidean gradient Z onto the tangent space at X of the manifold of rank-r tensors
    Implements 'Algorithm 2' of Steinlechner 2016 {doi.org/10.1137/15M1010506}

    :param U:               The left orthogonal cores of X (must have type TTCore)
    :param V:               The right orthogonal cores of X (must have type TTCore)
    :param X:               The current tensor in TT format
    :param A_vec:           Vector of data values at the sampling points
    :param Omega_indices:   The indices of the sampling points in X and the data tensor A. Formatted as array of size |Omega| x d
    :return:                List of first order variations describing the Riemannian gradient. Each variation has type TTCore
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
        Z_val = X[idx] - A_vec[count]
        V_product = np.array([[1]])
        for mu in range(d - 1, 0, -1):
            C[mu][:, idx[mu], :] += U_products[mu - 1].T * V_product.T * Z_val
            V_product = V[mu][:, idx[mu], :] @ V_product
        C[0][:, idx[0], :] += V_product.T * Z_val

    # Find the first order variations by projecting the C tensors (appart from the last one) onto the orthogonal compliment of the range of the associated U.L
    # as U is known to be left orthogonal for mu up to and including d-1, U.L is already an orthonormal matrix, so no explicit QR decomposition is necessary
    dU = []
    for mu in range(d - 1):
        C_local = TTCore(C[mu])
        dU.append(C_local.from_L(C_local.L - U[mu].L @ (U[mu].L.T @ C_local.L)))  # Smart matrix product factorization allows lowering the cost by a factor n
    dU.append(TTCore(C[-1]))  # The last variation is simply the C tensor

    # Return the first order variations
    return dU


def tangent_space_vector_transport(U, V, W):
    """
    Transports a tangent tensor Y with TT cores W into the tangent space of the rank-r manifold at X, where X has left-orthogonal cores U and right-orthogonal cores V
    Based on Steinlechner 2016 {doi.org/10.1137/15M1010506} pg 11

    :param U:   The left orthogonal cores of X (must be an iterable of TTCores)
    :param V:   The right orthogonal cores of X (must be an iterable of TTCores)
    :param W:   The cores of the tangent tensor Y (must have type TTCore)
    :return:    List of first order variations describing the projection of Y onto the tangent space at X
    """
    # Get dimension of X
    d = len(U)

    # Pre-compute the Y^T_{geq mu+1}X_{geq mu+1} terms, here called YTX
    YTX = [np.array([1])]
    for mu in range(d - 1, 0, -1):
        YTX = [W[mu].R @ spsp.kron(YTX[0], spsp.eye(W[mu].shape[1])) @ V[mu].R.T] + YTX

    # Iterate upwards through mu and compute the first order variations dU
    # Iteratively compute the (I_{n_mu} kronecker X_{leq mu-1})^T Y_{leq mu} terms, here called IXTY, on the fly
    dU = []
    IXTY = W[0].L
    for mu in range(d):
        dU_L = IXTY @ YTX[mu]

        if mu < d - 1:
            # All but the last variation must be projected onto the orthogonal compliment of the range of the associated U.L
            # as U is known to be left orthogonal for mu up to and including d-1, U.L is already an orthonormal matrix, so no explicit QR decomposition is necessary
            dU_L = dU_L - U[mu].L @ (U[mu].L.T @ dU_L)  # Smart matrix product factorization allows lowering the cost by a factor n

            # Also update IXTY
            IXTY = spsp.kron(spsp.eye(W[mu + 1].shape[1]), U[mu].L.T @ IXTY) @ W[mu + 1].L

        if len(dU_L.shape) == 1:
            dU_L = dU_L.reshape(-1, 1)
        dU.append(U[mu].from_L(dU_L))

    # Return the first order variations
    return dU


def vars_to_TT(U, V, dU) -> TT:
    """
    Turns a list of first order variations in the tangent space of X into a TT tensor

    :param U:   The left orthogonal cores of X (must be an iterable of TTCores)
    :param V:   The right orthogonal cores of X (must be an iterable of TTCores)
    :param dU:  The first order variations describing a direction in the tangent space of X
    :return:    The TT tensor decribed by the first order variations
    """
    # Implements formulation 3.11 from Steinlechner 2016 {doi.org/10.1137/15M1010506}
    new_cores = [TTCore(np.concatenate((np.array(dU[0]), np.array(U[0])), axis=-1))]
    for i in range(1, len(U) - 1):
        top_row = np.concatenate((np.array(V[i]), np.zeros((V[i].shape[0], V[i].shape[1], U[i].shape[-1]))), axis=-1)
        bottom_row = np.concatenate((np.array(dU[i]), np.array(U[i])), axis=-1)
        new_cores.append(TTCore(np.concatenate((top_row, bottom_row), axis=0)))

    new_cores.append(TTCore(np.concatenate((np.array(V[-1]), np.array(dU[-1])), axis=0)))

    return TT(new_cores)


def update_X(U, V, alpha, dU) -> TT:
    """
    Gives iteration X^{i+1} based on the cores of X^{i}, the step size, and the direction

    :param U:       The left orthogonal cores of X (must an iterable of TTCores)
    :param V:       The right orthogonal cores of X (must an iterable of TTCores)
    :param alpha:   The step size
    :param dU:      The first order variations describing the direction
    :return:        The TT tensor decribed by the first order variations
    """
    # Implements the corrected version of the formulation on Steinlechner 2016 {doi.org/10.1137/15M1010506} pg 11. The first term on the RHS should have dU1 instead of V1
    # First scale the variations by the step size and add the last left-orthogonal core to the last scaled variation
    for i in range(len(dU)):
        dU[i] *= alpha

    dU[-1] += U[-1]

    # Then simply use the vars_to_TT function
    return vars_to_TT(U, V, dU)


def inner_product(X: TT, Y: TT) -> float:
    """
    Calculates the inner product of two tensors in the TT format.
    Based on the approach implemented in the reference Matlab implementation provided for Steinlechner 2016 {doi.org/10.1137/15M1010506}
    :param X: Tensor 1 in TT format
    :param Y: Tensor 2 in TT format
    :return: inner product as a float
    """

    if X.d != Y.d:
        raise Exception("The two tensors must have the same order")

    res = np.array([[1]])
    for i in range(X.d):
        temp = X.cores[i].from_R(res.T @ X.cores[i].R)
        res = temp.L.T @ Y.cores[i].L

    return float(res)
