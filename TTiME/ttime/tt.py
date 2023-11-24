import os
import warnings
import numpy as np
from itertools import product
from tqdm import tqdm
from copy import deepcopy
from .tt_core import TTCore
from .tensor import Tensor


class TT:
    """
    Class for the Tensor Train (TT) representation of a tensor
    """
    def __init__(self, cores):
        """
        Initialize the TT using an iterable of cores. If one or multiple cores are not TTCore instances, they will be turned into such.
        Each core must be 3d and the third and first dimensions of neighbouring cores must match in size
        """
        self.cores = list(cores)
        self.__orthogonal_mu = None  # When orthogonalizing the tensor, this will indicate w.r.t. which core the tensor is orthogonalized. Uses the convention of Oseledets 2011 {DOI: 10.1137/090752286}, so the range is [1, d]
        self.__orthogonal_comparison_cores = None  # Stores the up-to-date cores after each orthogonalization. Allows checking whether self.__orthogonal_mu is still up to date
        for i in range(len(cores)):
            # Check the dimension of each core
            if not self.cores[i].ndim == 3:
                raise Exception("All input cores must be third order tensors")

            # Check that the first and last cores satisfy r_0 = r_d = 1
            if i == 0 and self.cores[i].shape[0] != 1:
                raise Exception("The first core must have size 1 along dimension 0")

            if i == len(cores) - 1 and self.cores[i].shape[-1] != 1:
                raise Exception("The last core must have size 1 along dimension 2")

            # Check the size compatibility of the cores
            if i > 0:
                if not self.cores[i].shape[0] == self.cores[i - 1].shape[-1]:
                    raise Exception(f"The cores at positions {i - 1} and {i} have incompatible shapes")

            # If applicable, turn entries into TTCore instances
            if not isinstance(self.cores[i], TTCore):
                self.cores[i] = TTCore(self.cores[i])

    def __getitem__(self, item):
        """
        The TT format is designed for easy access of individual entries but not for slices. Thus slicing is not suppoerted
        """
        if len(item) != self.d or not all(np.issubdtype(type(idx), np.integer) for idx in item):
            raise Exception("Only individual elements, not slices, can be accessed. Indices must be integers")

        result = self.cores[0][:, item[0], :]
        for i in range(1, len(item)):
            result = result @ self.cores[i][:, item[i], :]

        return result[0, 0]

    @property
    def d(self):
        """
        The dimension of a TT tensor equals its number of cores
        """
        return len(self.cores)

    @property
    def r(self):
        """
        The first and last rank are always 1, the other values depend on the sizes of the cores
        """
        r = np.ones(self.d + 1, dtype=int)
        for i in range(1, self.d):
            r[i] = self.cores[i].shape[0]
            if r[i] != self.cores[i - 1].shape[-1]:
                raise Exception(f"The cores at positions {i - 1} and {i} have incompatible shapes")

        if not self.cores[0].shape[0] == self.cores[-1].shape[-1] == 1:
            raise Exception('The first and last cores must be such that r[0] and r[-1] equal 1')

        return r

    @property
    def n(self):
        """
        Gives the size of the tensor along the respective dimensions. This information is stored in the size of each core along dimension 1
        """
        n = np.zeros(self.d, dtype=int)
        for i in range(self.d):
            n[i] = self.cores[i].shape[1]

        return n

    @property
    def norm(self):
        """
        Uses the properties of the TT format to efficiently compute the norm of the tensor
        """
        if not self.__check_orthogonality():
            # The TT is not orthogonal so we need to orthogonalize it. The choice of orthogonalization mode is arbitrary
            self.left_orthogonalize()
            return np.linalg.norm(self.cores[-1])  # last core as we left-orthogonalized

        # Else the TT is already orthogonal with known mode
        return np.linalg.norm(self.cores[self.__orthogonal_mu - 1])

    @property
    def full_tensor(self):
        """
        The TT format is made for very large tensors. Direct access of the full uncompressed tensor can be very costly and memory consuming.
        Thus, an error or warning is thrown if the full tensor has more than a million cells
        """
        if np.prod(self.n) > 1e5:
            raise Exception("The full tensor has more than 1e5 entries. To nonetheless compute it, use obj.get_fulltensor(size_error=False)")

        return self.get_full_tensor()

    def get_full_tensor(self, size_error=True):
        """
        Using this method allows de-activting the size error and thus accessing very large tensors
        """
        if np.prod(self.n) > 1e5:
            msg = "The full tensor has more than 1e5 entries"
            if size_error:
                raise Exception(msg + '. To nonetheless compute it, set size_error=False')
            else:
                warnings.warn(msg)

        tensor = np.zeros(self.n)
        idcs = list(product(*(range(i) for i in self.n)))
        for i in tqdm(range(len(idcs)), disable=(np.prod(self.n) <= 1e4)):
            # We only want a progress bar when the output tensor is large
            tensor[idcs[i]] = self.__getitem__(idcs[i])

        return Tensor(tensor)

    def mu_orthogonalize(self, mu):
        if not (isinstance(mu, int) and 1 <= mu <= self.d):
            raise Exception(f"mu must be an integer in the range [1, {self.d}]")

        # First go from the left
        for i in range(0, mu - 1):
            Q, R = np.linalg.qr(self.cores[i].L)
            self.cores[i] = self.cores[i].from_L(Q)
            self.cores[i + 1] = self.cores[i + 1].from_R(R @ self.cores[i + 1].R)

        # Then from the right
        for i in range(self.d - 1, mu - 1, -1):
            Q, R = np.linalg.qr(self.cores[i].R.T)
            self.cores[i] = self.cores[i].from_R(Q.T)
            self.cores[i - 1] = self.cores[i - 1].from_L(self.cores[i - 1].L @ R.T)

        # With the orthogonalization complete, store the orthogonalization parameters for future reference
        self.__orthogonal_mu = mu
        self.__orthogonal_comparison_cores = deepcopy(self.cores)  # Important to copy, not just reference, the cores

    def left_orthogonalize(self):
        """
        The TT is left-orthogonal if it is d-orthogonal
        """
        self.mu_orthogonalize(self.d)

    def right_orthogonalize(self):
        """
        The TT is right-orthogonal if it is 1-orthogonal
        """
        self.mu_orthogonalize(1)

    def __check_orthogonality(self):
        """
        returns True if the current cores are orthogonal, False otherwise
        """

        if self.__orthogonal_mu is None:
            return False

        for i in range(self.d):
            if not np.array_equal(self.cores[i], self.__orthogonal_comparison_cores[i], equal_nan=True):
                self.__orthogonal_mu = None  # Make future checks cheaper
                return False

        return True

    def truncate_rank(self, new_r, mu=None):
        """
        Truncate the TT rank to the desired value. If new_r is an iterable, it must be of length self.d - 1 (the 1s at the ends do not need to be included) and mu should equal None.
        If new_r is a single integer, mu must be specified, because the rank between the cores with indices mu - 1 and mu will be reduced
        """
        if hasattr(new_r, '__iter__'):
            if len(new_r) != self.d - 1:
                raise Exception(f"new_r must be an integer or an iterable of integers of length {self.d - 1} (do not include the leading and trailing 1s of the rank tuple))")

            if not all(np.issubdtype(type(r), np.integer) for r in new_r):
                raise Exception("all elements of new_r must be integers")

            if any(new_r[i] > self.r[i + 1] or new_r[i] < 1 for i in range(len(new_r))):
                raise Exception("new rank values cannot be zero, negative, or greater than the current ones")

            if mu is not None:
                warnings.warn('setting mu when new_r is an iterable has no effect')

            # If the TT is not yet so, we must left-orthogonalize it
            if not (self.__orthogonal_mu != self.d and self.__check_orthogonality()):
                self.left_orthogonalize()

            # Then truncate the ranks, going from right to left
            for mu in range(self.d - 1, 0, -1):
                self.__truncate_one_rank(new_r[mu - 1], mu)

            # The tensor is now known to be right-orthogonal
            self.__orthogonal_mu = 1
            self.__orthogonal_comparison_cores = deepcopy(self.cores)

        else:
            if not np.issubdtype(type(new_r), np.integer):
                raise Exception(f"new_r must be an integer or an iterable of integers of length {self.d - 1}")

            if mu is None:
                raise Exception("When new_r is a single integer, mu must be an integer as well")

            if not (np.issubdtype(type(mu), np.integer) and 1 <= mu <= self.d - 1):
                raise Exception(f"mu must be an integer in the range [1, {self.d - 1}]")

            if new_r > self.r[mu] or new_r < 1:
                raise Exception("new_r values cannot be zero, negative, or greater than the current ones")

            # If the TT is not yet so, we must (mu +1)-orthogonalize it
            if not (self.__orthogonal_mu != mu + 1 and self.__check_orthogonality()):
                self.mu_orthogonalize(mu + 1)

            # Then truncate the indicated rank
            self.__truncate_one_rank(new_r, mu)

            # The tensor is now known to be mu-orthogonal
            self.__orthogonal_mu = mu
            self.__orthogonal_comparison_cores = deepcopy(self.cores)

    def __truncate_one_rank(self, new_r, mu):
        """
        Truncate the mu^th rank (index mu in the rank array, located between cores with indices mu-1 and mu).
        It is assumed that the TT is known to be mu + 1 orthogonal
        """
        # Perform the SVD
        Q, S, VT = np.linalg.svd(self.cores[mu].R)

        # Keep only the first 'new_r' singular values
        Q = Q[:, :new_r]
        S = np.diag(S[:new_r])
        VT = VT[:new_r]

        # Now adjust cores with indices mu and mu - 1. The dimensions have changed so we cannot simply assign to the unfoldings
        self.cores[mu] = self.cores[mu].from_R(VT)
        self.cores[mu - 1] = self.cores[mu - 1].from_L(self.cores[mu - 1].L @ Q @ S)

    def random_vec_rank_increase(self, mu, magnitude=1e-8):
        """
        As explained by Steinlechner 2016 {doi.org/10.1137/15M1010506} pg 14, the rank between two cores can be increased by one
        by concatenating two random vectors to the left and right unfoldings of the cores with indices mu-1 and mu respectively
        """
        if not (np.issubdtype(type(mu), np.integer) and 1 <= mu <= self.d - 1):
            raise Exception(f"mu must be an integer in the range [1, {self.d - 1}]")

        random_column_vec = np.random.random((self.cores[mu - 1].n[:-1].prod(), 1))
        random_column_vec *= magnitude / np.linalg.norm(random_column_vec)
        self.cores[mu - 1] = self.cores[mu - 1].from_L(np.hstack((self.cores[mu - 1].L, random_column_vec)))

        random_row_vec = np.random.random((1, self.cores[mu].n[1:].prod()))
        random_row_vec *= magnitude / np.linalg.norm(random_row_vec)
        self.cores[mu] = self.cores[mu].from_R(np.vstack((self.cores[mu].R, random_row_vec)))

    def mode_mu_multiply(self, A, mu):
        """
        Do the mode-mu multiplication of the TT tensor with a matrix A
        Based on Glau 2020 {doi.org/10.1137/19M1244172}
        """

        if not (np.issubdtype(type(mu), np.integer) and 1 <= mu <= self.d):
            raise Exception(f"mu must be an integer in the range [1, {self.d}]")

        if not ((isinstance(A, np.ndarray) or isinstance(A, Tensor)) and A.ndim == 2):
            raise Exception("A must be a 2D Tensor object or numpy array")

        if not A.shape[1] == self.cores[mu - 1].shape[1]:
            raise Exception("Dimension 1 of A must match dimension 1 of the TT core with index mu-1")

        shape = self.cores[mu - 1].shape
        temp = A @ self.cores[mu - 1].transpose((1, 0, 2)).reshape((shape[1], shape[0] * shape[2]), order='F')
        self.cores[mu - 1] = temp.reshape((A.shape[0], shape[0], shape[2]), order='F').transpose((1, 0, 2))

    def save_cores(self, directory):
        """
        Saves the left unfoldings of the cores as text files to the directory specified by 'directory'.
        Each core gets its own file, with file_names following the convention 'coreL_i.txt', with i from 0 to d-1
        """
        for i, core in enumerate(self.cores):
            np.savetxt(os.path.join(directory, f'coreL_{i}.txt'), core.L)
