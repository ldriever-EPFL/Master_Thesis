import numpy as np
from os.path import join
from .cheb_tt import ChebTT
from .tt import TT


class preChebTT(ChebTT):
    """
    Can be used like the ChebTT class, but in the case of pre_computed cores
    """

    def __init__(self, core_directory, intervals):
        """
        We overload the init function as we now only have to read in the cores, not create them using expensive fucntion evaluations

        :param core_directory:  Path to the directory where the cores are stored. If ChebTT was used to create the cores, this should
                                be "path_to_Storage_directory/Final_cores". It is assumed that the original naming of the core files
                                (i.e. coreL_{i}.txt) is used
        :param intervals:       Either the same iterable as would be passed to ChebTT, or a file path for a file containing said iterable
        """

        # Handle the inputs
        if type(intervals) == str:
            self.intervals = np.loadtxt(intervals)
        else:
            self.intervals = np.array(intervals)
        self.d = len(self.intervals)

        # Create the affine transforms for mapping the intervals to the basic [-1, 1] interval used for Chebyshev interpolations
        self.a_transforms_to_Cheb = []
        for i in range(self.d):
            self.a_transforms_to_Cheb.append(lambda x, low=self.intervals[i][0], high=self.intervals[i][1]: (x - low) / (high - low) * 2 - 1)

        # Load the core left unfoldings
        coreL_storage = []
        for i in range(self.d):
            try:
                coreL_storage.append(np.loadtxt(join(core_directory, f'coreL_{i}.txt')))
            except FileNotFoundError:
                raise Exception(f"The file containing the left unfolding of core {i} could not be found. Check the file naming and that the number of specified intervals matches the number of precomputed cores")

        # Extract the rank tuple, the tensor dimension, and the reshaped cores
        self.r = [1]
        self.n = []
        cores = []
        for coreL in coreL_storage:
            self.n.append(coreL.shape[0] // self.r[-1])
            try:
                self.r.append(coreL.shape[1])
            except IndexError:
                # Numpy reads the last unfolded core as a 1D array so special handling is required
                self.r.append(1)
            cores.append(coreL.reshape((self.r[-2], self.n[-1], self.r[-1]), order='F'))

        self.r = np.array(self.r)
        self.n = np.array(self.n)

        # Now we can also extract the orders and create the P TT object
        self.orders = self.n - 1
        self.P = TT(cores)

        # Finally compute the C TT
        self._ChebTT__compute_C()

