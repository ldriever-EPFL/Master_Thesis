import os
import sys
import unittest
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from ttime import RTTC, TT
from ttime.rttc_helper import inner_product


class TestRTTC(unittest.TestCase):
    """
    Testing functions related to the RTTC class in rttc.py
    """

    def test_inner_product(self):
        """
        Tests the inner_product(X, Y) function in rttc_helper.py
        """

        X = TT([np.random.random((1, 10, 3)), np.random.random((3, 7, 4)), np.random.random((4, 7, 1))])
        Y = TT([np.random.random((1, 10, 3)), np.random.random((3, 7, 4)), np.random.random((4, 7, 1))])

        expected_val = float(np.dot(X.full_tensor.vec.flatten(), Y.full_tensor.vec.flatten()))

        self.assertAlmostEqual(expected_val, inner_product(X, Y), msg='Inner product is incorrect')

