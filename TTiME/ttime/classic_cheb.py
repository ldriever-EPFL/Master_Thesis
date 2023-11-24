import numba
import numpy as np
from .tensor import Tensor


@numba.jit(nopython=True, cache=True)
def idx_val_to_array(idx, d, n_cumprod):
    """
    Turns the enumeration of a tensor's entries into tuples of indices along each dimension
    """
    idx_tuple_array = np.zeros(d, dtype=np.int64)
    for i in range(d):
        idx_tuple_array[i] = idx // n_cumprod[-(1 + i)]
        idx = idx % n_cumprod[-(1 + i)]

    return idx_tuple_array


@numba.jit(nopython=True, cache=True)
def get_c_val(idx_array: np.ndarray, P: np.ndarray, n_cumprod: np.ndarray) -> float:
    """
    Combutes a single Chebyshev interpolation coefficient
    """
    P_vec = P.flatten()
    N = np.array(P.shape) - 1  # interpolation order
    factor = np.prod(2 ** ((0 < idx_array) & (idx_array < N)) / N)

    summand = 0.
    for i in range(np.prod(np.array(P.shape))):
        local_idx_array = idx_val_to_array(i, P.ndim, n_cumprod)
        local_factor = 0.5 ** np.sum((local_idx_array == 0) | (local_idx_array == N))
        local_product = np.prod(np.cos(idx_array * np.pi * local_idx_array / N))

        summand = summand + local_factor * P_vec[i] * local_product

    return factor * summand


@numba.jit(nopython=True, cache=True)
def compute_C_tensor(P: np.ndarray) -> np.ndarray:
    """
    Computes the tensor of Chebyshev interpolation coefficients
    """
    C = np.zeros(P.shape)
    C = C.flatten()
    n_cumprod = np.cumprod(np.concatenate((np.array([1]), np.array(P.shape[:-1]))))

    for i in range(np.prod(np.array(P.shape))):
        idx_tuple = idx_val_to_array(i, P.ndim, n_cumprod)
        C[i] = get_c_val(idx_tuple, P, n_cumprod)

    return C.reshape(P.shape)


@numba.jit(nopython=True, cache=True)
def get_point(cheb_inputs: np.ndarray, C: np.ndarray) -> float:
    """
    Returns the value of the Chebyshev interpolation at a certain point
    """
    n_cumprod = np.cumprod(np.concatenate((np.array([1]), np.array(C.shape[:-1]))))
    C_vec = C.flatten()
    total = 0.
    for i in range(np.prod(np.array(C.shape))):
        local_idx_array = idx_val_to_array(i, C.ndim, n_cumprod)
        total += C_vec[i] * np.prod(np.cos(local_idx_array * np.arccos(cheb_inputs)))

    return total


@numba.jit(nopython=True, cache=True)
def get_one_axis_derivative(cheb_inputs: np.ndarray, C: np.ndarray, ax: float) -> float:
    """
    Computes the derivative of the Chebyshev interpolation at a certain point along the specified axis
    """
    n_cumprod = np.cumprod(np.concatenate((np.array([1]), np.array(C.shape[:-1]))))
    C_vec = C.flatten()
    total = 0.
    for i in range(np.prod(np.array(C.shape))):
        local_idx_array = idx_val_to_array(i, C.ndim, n_cumprod)
        # Derive the Chebyshev polynomial with index 'ax'

        if cheb_inputs[ax] == 1:
            dif = local_idx_array[ax] ** 2
        elif cheb_inputs[ax] == -1:
            if local_idx_array[ax] % 2 == 0:
                # even case
                dif = - local_idx_array[ax] ** 2
            else:
                # odd case
                dif = local_idx_array[ax] ** 2
        else:
            dif = local_idx_array[ax] * np.sin(local_idx_array[ax] * np.arccos(cheb_inputs[ax])) / np.sqrt(1 - cheb_inputs[ax] ** 2)

        if ax == 0:
            total += C_vec[i] * dif * np.prod(np.cos(local_idx_array[ax + 1:] * np.arccos(cheb_inputs[ax + 1:])))
        elif ax == len(cheb_inputs) - 1:
            total += C_vec[i] * np.prod(np.cos(local_idx_array[:ax] * np.arccos(cheb_inputs[:ax]))) * dif
        else:
            total += C_vec[i] * np.prod(np.cos(local_idx_array[:ax] * np.arccos(cheb_inputs[:ax]))) * dif * np.prod(np.cos(local_idx_array[ax + 1:] * np.arccos(cheb_inputs[ax + 1:])))

    return total


class ClassicCheb:

    """
    Class for tensorized Chebyshev interpolation without low-rank approximation
    """

    def __init__(self, inputs, data_vec, intervals, order, input_type, P=None, C=None):
        """
        Turns a collection of data points, which describe a function's value at all points on a certain Chebyshev grid, into the corresponding Chebyshev interpolation.
        Uses the concepts presented in Glau 2020 {doi.org/10.1137/19M1244172}

        :param inputs:      provides the 'location' of each tensor entry, either as index tuples, points in the original domain, or points in the transformed [-1, 1] Chebyshev interval. Must have as many entries as the tensor
        :param data_vec:    provides the function value corresponding to each tensor entry indicated in 'inputs'. Must have as many entries as the tensor
        :param order:       order of the Chebysehv interpolation
        :param input_type:  'index', 'real', or 'Cheb', depending on how the tensor entry locations are specified in 'inputs'
        :param P:           optional. Allows loading a pre-computed P tensor, avoiding the need to re-compute it
        :param C:           optional. Allows loading a pre-computed C tensor, avoiding the need to re-compute it. Can lead to great computational savings
        """

        self.intervals = intervals
        self.transforms_to_Cheb = self.__get_a_transforms_to_Cheb()
        self.d = len(intervals)

        if np.size(order) == 1:
            order = self.d * [order]
        self.order = np.array(order)

        # Transform the inputs to indices in the P tensor
        if input_type == 'index':
            # Inputs already give indices in the P tensor
            P_indices = inputs
        else:
            if input_type == 'real':
                # Inputs reflect those passed to the to-be-interpolated function
                Cheb_coords = [self.__get_Cheb_inputs(point) for point in inputs]
            elif input_type == 'Cheb':
                # Inputs are already transormed to the [-1, 1] interval
                Cheb_coords = inputs
            else:
                raise Exception("input type must be one of ['real', 'Cheb', 'index]")

            P_indices = np.round(np.arccos(Cheb_coords) / np.pi * order).astype(int)

        P_indices = tuple(map(tuple, P_indices))

        # Create the P-tensor
        if P is None:
            P = Tensor(np.zeros(self.order + 1))
            for i in range(P.size):
                P[P_indices[i]] = data_vec[i]
        self.P = P

        # Compute the C tensor
        if C is None:
            C = compute_C_tensor(np.asarray(self.P))
        self.C = C

    def __get_a_transforms_to_Cheb(self):
        a_transforms_to_Cheb = []
        for i in range(len(self.intervals)):
            a_transforms_to_Cheb.append(lambda x, low=self.intervals[i][0], high=self.intervals[i][1]: (x - low) / (high - low) * 2 - 1)

        return a_transforms_to_Cheb

    def __get_Cheb_inputs(self, point):
        point = np.ravel(point)
        Cheb_inputs = []
        for i, val in enumerate(point):
            if not self.intervals[i][0] <= val <= self.intervals[i][1]:
                raise Exception(f'Provided value is outside the interpolation interval along axis {i}')

            Cheb_inputs.append(self.transforms_to_Cheb[i](val))

        return np.array(Cheb_inputs)

    def __getitem__(self, item):
        """
        Returns the value of the Chebyshev interpolation for the specified input point
        """
        return get_point(self.__get_Cheb_inputs(item), self.C)

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
            derivative.append(get_one_axis_derivative(Cheb_inputs, self.C, ax))
            derivative[-1] *= 2 / (self.intervals[ax, 1] - self.intervals[ax, 0])  # Account for the derivative of the affine transform to the Chebyshev domain

        return np.array(derivative)
