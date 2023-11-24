import numpy as np
from .numpy_inheritance_helper import adapt_array_ufunc_part_1, adapt_array_ufunc_part_2, adapt_array_function


class Tensor(np.ndarray):
    """
    A wrapper for numpy.ndarray class, which provides all the methods accessible for a numpy array,
    but also provides additional functionalities specific to tensors.
    Inspired by Steinlechner 2016 {doi.org/10.1137/15M1010506}
    """

    def __new__(cls, input_array):
        # Create a new instance of the custom array, based on the np.ndarray class
        if np.issubdtype(input_array.dtype, np.integer) or np.issubdtype(input_array.dtype, np.floating):
            input_array = input_array.astype(np.float64)
        obj = np.asarray(input_array).view(cls)
        return obj

    def __init__(self, input_array):
        self.__r = None  # The TT rank is initially not computed

    def __getitem__(self, index):
        # Ensures that a sliced Tensor is still a Tensor
        return Tensor(super(Tensor, self).__getitem__(index))

    def __setitem__(self, index, value):
        # The tensor content changes so its TT rank will have to be recomputed
        self.__r = None

        # then perform the indexing operation as usual
        self.data[index] = value

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # Ensure that numpy ufuncs return an instance of the Tensor class when operating on such an instance
        args, kwargs, outputs = adapt_array_ufunc_part_1(Tensor, ufunc, *inputs, out=None, **kwargs)
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        return adapt_array_ufunc_part_2(Tensor, Tensor, ufunc, results, outputs)

    def __array_function__(self, func, types, *args, **kwargs):
        return adapt_array_function(Tensor, Tensor, func, *args, **kwargs)

    @property
    def d(self):
        # For more consistent naming between the class and the reference papers
        return self.ndim

    @property
    def n(self):
        # Get the shape as an array, for easier handling
        # gives the number of points along each dimension
        return np.array(self.shape)

    @property
    def r(self):
        # If the TT rank has not been computed for the current state of the tensor, then do so
        if self.__r is None:
            self.__compute_TT_rank()

        # Then return the TT rank
        return self.__r

    @property
    def vec(self):
        # Return the tensor as a column vector. Same effect as calling unfold(d)
        return self.reshape(-1, 1, order='F')

    def unfold(self, mode):
        # Based on the MATLAB-like unfolding method presented in Oseledets 2011 {DOI: 10.1137/090752286}
        # Method maintains the original nomenclature, such that, e.g., a mode-1 unfolding unfolds in dimension 0
        # mode-0 and mode-d return row and column vectors respectively
        if not isinstance(mode, int) or mode < 0 or mode > self.d:
            raise Exception(f'mode must be an integer in the interval [0, {self.d}]')

        # It is essential that, in line with Oseledets 2011 {DOI: 10.1137/090752286}, we use FORTRAN-contiguous order
        return np.ascontiguousarray(self.reshape((np.prod(self.n[:mode]), np.prod(self.n[mode:])), order='F'))

    def __compute_TT_rank(self):
        self.__r = np.ones(self.d + 1)  # ranks zero and d are equal to one
        for i in range(1, self.d):
            # fill positions 1 to d-1 with the ranks of the corresponding unfoldings
            self.__r[i] = np.linalg.matrix_rank(self.unfold(i))
