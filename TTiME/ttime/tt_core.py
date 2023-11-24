import numpy as np
from .tensor import Tensor
from .numpy_inheritance_helper import adapt_array_ufunc_part_1, adapt_array_ufunc_part_2, adapt_array_function


class TTCore(Tensor):
    """
    A Tensor instance, which is to act as the core of a tensor train (see the TT class) has certain methods applicable to it
    """

    def __init__(self, input_array):
        super(TTCore, self).__init__(input_array)

        if self.d != 3:
            raise Exception('A TTCore must be a third order tensor')

    def __getitem__(self, index):
        # Ensures that a sliced TTCore is a TTCore if it has dimension 3 and is a Tensor otherwise
        result = super(TTCore, self).__getitem__(index)  # returns a Tensor object
        if result.d == 3:
            return TTCore(result)
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # Ensure that numpy ufuncs return an instance of the Tensor class when operating on such an instance
        args, kwargs, outputs = adapt_array_ufunc_part_1(TTCore, ufunc, *inputs, out=None, **kwargs)
        results = super(Tensor, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        return adapt_array_ufunc_part_2(TTCore, Tensor, ufunc, results, outputs)

    def __array_function__(self, func, types, *args, **kwargs):
        return adapt_array_function(TTCore, Tensor, func, *args, **kwargs)

    @property
    def L(self):
        # A quick way to access the left unfolding, which is simply the mode-2 unfolding of a TT core
        return self.unfold(2)

    @L.setter
    def L(self, matrix):
        # Make it possible to modify the TTCore by modifying its left unfolding. If matrix is of a different shape than self.L, use self.from_L(matrix) instead
        # Check that value has an allowed shape:
        if matrix.shape != self.L.shape:
            raise Exception("When assigning new values to the left unfolding using the = operator, the new value must be a matrix of equal shape to the left unfolding.\nIf the number of columns changed use obj = obj.from_L(matrix) instead")

        # Ensure that the input is of the correct datatype
        if np.issubdtype(matrix.dtype, np.integer) or np.issubdtype(matrix.dtype, np.floating):
            matrix = matrix.astype(np.float64)
        else:
            raise Exception('matrix must have integer or float as dtype')

        matrix = np.ascontiguousarray(matrix.reshape(self.n, order='F'))
        self.data = matrix
        self.strides = matrix.strides

    def from_L(self, matrix):
        # In case that the shape of the left unfolding is changed, its memory size is no longer compatible with the current TTCore instance
        # This method returns the TTCore that is related to the provided matrix and the shape of the current TTCore instance
        # Use as: core = core.from_L(matrix)

        # Check that value has an allowed shape:
        if matrix.ndim != 2 or matrix.shape[0] != self.n[0] * self.n[1]:
            raise Exception("When getting a TTCore from its left unfolding, the new value has to be a matrix with an equal number of rows as the left unfolding")

        # Fold the matrix to the intended shape and return as TTCore
        return TTCore(np.ascontiguousarray(matrix.reshape((self.n[0], self.n[1], matrix.shape[1]), order='F')))

    @property
    def R(self):
        # A quick way to access the right unfolding, which is simply the mode-1 unfolding of a TT core
        return self.unfold(1)

    @R.setter
    def R(self, matrix):
        # Make it possible to modify the TTCore by modifying its right unfolding. If matrix is of a different shape than self.R, use self.from_R(matrix) instead
        # Check that value has an allowed shape:
        if matrix.shape != self.R.shape:
            raise Exception("When assigning new values to the right unfolding using the = operator, the new value must be a matrix of equal shape to the right unfolding.\nIf the number of rows changed use obj = obj.from_R(matrix) instead")

        # Ensure that the input is of the correct datatype
        if np.issubdtype(matrix.dtype, np.integer) or np.issubdtype(matrix.dtype, np.floating):
            matrix = matrix.astype(np.float64)
        else:
            raise Exception('matrix must have integer or float as dtype')

        matrix = np.ascontiguousarray(matrix.reshape(self.n, order='F'))
        self.data = matrix
        self.strides = matrix.strides

    def from_R(self, matrix):
        # In case that the shape of the right unfolding is changed, its memory size is no longer compatible with the current TTCore instance
        # This method returns the TTCore that is related to the provided matrix and the shape of the current TTCore instance
        # Use as: core = core.from_L(matrix)

        # Check that value has an allowed shape:
        if matrix.ndim != 2 or matrix.shape[1] != self.n[1] * self.n[2]:
            raise Exception("When getting a TTCore from its right unfolding, the new value has to be a matrix with an equal number of columns as the right unfolding")

        # Fold the matrix to the intended shape and return as TTCore
        return TTCore(np.ascontiguousarray(matrix.reshape((matrix.shape[0], self.n[1], self.n[2]), order='F')))
