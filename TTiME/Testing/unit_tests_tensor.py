import os
import sys
import unittest
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from ttime import Tensor


class TestTensor(unittest.TestCase):
    """
    Testing the Tensor class in tensor.py
    """

    def setUp(self):
        np.random.seed(99)
        self.test_tensor = Tensor(np.random.randint(0, 100, (3, 2, 4, 2)))

    def test_geometric_properties(self):
        self.assertEqual(4, self.test_tensor.d, msg='Tensor does not have the expected dimension')
        np.testing.assert_array_equal((3, 2, 4, 2), self.test_tensor.n, err_msg='Tensor does not have the expected shape')
        self.assertEqual(np.ndarray, type(self.test_tensor.n), msg='Tensor shape is not returned as Numpy array')

    def test_vectorization(self):
        """
        Test the vectorization format against the result expected from vectorizing the tensor in MATLAB, in line with the convention of doi.org/10.1137/090752286
        """
        expected_vec = [[1], [59], [73], [52], [59], [50], [57], [2], [21], [23], [79], [5], [73], [12], [99], [55], [27], [96], [68], [72], [22], [48], [62], [91], [35], [87], [32], [1], [39], [59], [40], [64], [41], [35], [47], [12], [82], [40], [19], [65], [62], [53], [69], [20], [26], [93], [17], [1]]
        np.testing.assert_array_equal(expected_vec, self.test_tensor.vec, err_msg='Vectorization of tensor has the wrong values or shape or both')

    def test_unfolding(self):
        """
        Test the unfolding result against the result of using the reshape function in MATLAB, in line with the convention of doi.org/10.1137/090752286
        """
        # Test that the expected excpetion is thrown for invoalid input modes
        with self.assertRaises(Exception) as context:
            _ = self.test_tensor.unfold(2.3)
        self.assertEqual(str(context.exception), 'mode must be an integer in the interval [0, 4]')

        with self.assertRaises(Exception) as context:
            _ = self.test_tensor.unfold([1, 3])
        self.assertEqual(str(context.exception), 'mode must be an integer in the interval [0, 4]')

        with self.assertRaises(Exception) as context:
            _ = self.test_tensor.unfold(-2)
        self.assertEqual(str(context.exception), 'mode must be an integer in the interval [0, 4]')

        with self.assertRaises(Exception) as context:
            _ = self.test_tensor.unfold(5)
        self.assertEqual(str(context.exception), 'mode must be an integer in the interval [0, 4]')

        # All unfolding modes are tested. 0 and 4 should be equal and can be tested using the already tested Tensor.vec property
        expected_unfolding_0 = self.test_tensor.vec.T

        expected_unfolding_1 = [[1, 52, 57, 23, 73, 55, 68, 48, 35, 1, 40, 35, 82, 65, 69, 93],
                                [59, 59, 2, 79, 12, 27, 72, 62, 87, 39, 64, 47, 40, 62, 20, 17],
                                [73, 50, 21, 5, 99, 96, 22, 91, 32, 59, 41, 12, 19, 53, 26, 1]]

        expected_unfolding_2 = [[1, 57, 73, 68, 35, 40, 82, 69],
                                [59, 2, 12, 72, 87, 64, 40, 20],
                                [73, 21, 99, 22, 32, 41, 19, 26],
                                [52, 23, 55, 48, 1, 35, 65, 93],
                                [59, 79, 27, 62, 39, 47, 62, 17],
                                [50, 5, 96, 91, 59, 12, 53, 1]]

        expected_unfolding_3 = [[1, 35],
                                [59, 87],
                                [73, 32],
                                [52, 1],
                                [59, 39],
                                [50, 59],
                                [57, 40],
                                [2, 64],
                                [21, 41],
                                [23, 35],
                                [79, 47],
                                [5, 12],
                                [73, 82],
                                [12, 40],
                                [99, 19],
                                [55, 65],
                                [27, 62],
                                [96, 53],
                                [68, 69],
                                [72, 20],
                                [22, 26],
                                [48, 93],
                                [62, 17],
                                [91, 1]]

        expected_unfolding_4 = self.test_tensor.vec

        np.testing.assert_array_equal(expected_unfolding_0, self.test_tensor.unfold(0), err_msg='mode 0 unfolding is incorrect')
        np.testing.assert_array_equal(expected_unfolding_1, self.test_tensor.unfold(1), err_msg='mode 1 unfolding is incorrect')
        np.testing.assert_array_equal(expected_unfolding_2, self.test_tensor.unfold(2), err_msg='mode 2 unfolding is incorrect')
        np.testing.assert_array_equal(expected_unfolding_3, self.test_tensor.unfold(3), err_msg='mode 3 unfolding is incorrect')
        np.testing.assert_array_equal(expected_unfolding_4, self.test_tensor.unfold(4), err_msg='mode 4 unfolding is incorrect')

    def test_rank_computation_function(self):
        expected_r = [1, 3, 6, 2, 1]
        np.testing.assert_array_equal(expected_r, self.test_tensor.r)
        self.assertEqual(np.ndarray, type(self.test_tensor.r), msg="r is not a Numpy array")

    def test_rank_access_method(self):
        """
        We want to avoid recomputing r if it is accessed multiple times without changes being made to the tensor.
        Here we check that the provate attribute __r is reset to None if a change is made to the tensor
        """
        self.assertTrue(self.test_tensor._Tensor__r is None)
        _ = self.test_tensor.r
        self.assertTrue(self.test_tensor._Tensor__r is not None)
        _ = self.test_tensor.r
        self.assertTrue(self.test_tensor._Tensor__r is not None)
        self.test_tensor[0, 1, 0, 1] = 101  # change arbitrary element
        self.assertTrue(self.test_tensor._Tensor__r is None)
        _ = self.test_tensor.r
        self.assertTrue(self.test_tensor._Tensor__r is not None)
        self.test_tensor = self.test_tensor + 1
        self.assertTrue(self.test_tensor._Tensor__r is None)

    def test_slicing_object_type(self):
        """
        Ensure that slicing returns the expected result and that the slice is a Tensor object with all the typical attributes.
        This ensures that the slice is truly a Tensor instance, not just casted by name to that class
        """
        # Check indexed entry
        self.assertEqual(65, self.test_tensor[0, 1, 2, 1])

        # Check slice type
        sliced_tensor = self.test_tensor[:, :, 2:]
        self.assertEqual(Tensor, type(sliced_tensor), msg="Sliced tensor is not a member of the Tensor class")
        try:
            dunder_r = sliced_tensor._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for sliced tensor")
        except AttributeError as e:
            raise Exception(f"Slice is not a proper instance of the Tensor class. Initial error message: {e}")

    def test_ufunc(self):
        """
        Ensure that the np.ndarray ufuncs properly work on Tensor instances, and that the result too is a Tensor instance
        """
        # Test that np.sqrt gives the correct result and returns a Tensor instance. Again we double check the casting using _Tensor__r as test attribute
        sqrt_tensor = np.sqrt(self.test_tensor)
        self.assertEqual(8, sqrt_tensor[1, 0, 1, 1], msg='Square root result is not correct')

        self.assertEqual(Tensor, type(sqrt_tensor), msg="square rooted tensor is not a member of the Tensor class")
        try:
            dunder_r = sqrt_tensor._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for square rooted tensor")
        except AttributeError as e:
            raise Exception(f"Square rooted tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        # Test that '+' and '*' operators return a Tensor instance. Again we double check the casting using _Tensor__r as test attribute
        result_tensor = self.test_tensor * 2 + self.test_tensor

        self.assertEqual(Tensor, type(result_tensor), msg="result tensor is not a member of the Tensor class")
        try:
            dunder_r = result_tensor._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for result tensor")
        except AttributeError as e:
            raise Exception(f"result tensor is not a proper instance of the Tensor class. Initial error message: {e}")

    def test_general_func(self):
        """
        There are a lot of handy functions defined in other parts of the numpy package, and many will return ndarray instances.
        We check that, when using these functions on Tensor instances, the result too is a Tensor instance
        """
        # Test, using a 2d slice of the original tensor, that np.linalg.inv returns a tensor
        inv = np.linalg.inv(self.test_tensor[0, 0, :2, :])
        self.assertEqual(Tensor, type(inv), msg="inv is not a member of the Tensor class")
        try:
            dunder_r = inv._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for inv tensor")
        except AttributeError as e:
            raise Exception(f"inv tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        # Test that concatenating a Tensor to itself gives a Tensor of expected shape
        result = np.concatenate((self.test_tensor, self.test_tensor * 2), axis=3)
        self.assertEqual(self.test_tensor[2, 1, 3, 0] * 2, result[2, 1, 3, 2], msg="values in concatenated tensor are not as expected")
        self.assertEqual(Tensor, type(result), msg="result is not a member of the Tensor class")
        np.testing.assert_array_equal((3, 2, 4, 4), result.n)
        try:
            dunder_r = result._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for result tensor")
        except AttributeError as e:
            raise Exception(f"result tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        # Test that np.linalg.qr returns two tensors
        Q, R = np.linalg.qr(self.test_tensor)
        self.assertEqual(Tensor, type(Q), msg="Q tensor is not a member of the Tensor class")
        try:
            dunder_r = Q._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for Q tensor")
        except AttributeError as e:
            raise Exception(f"Q tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        self.assertEqual(Tensor, type(R), msg="R tensor is not a member of the Tensor class")
        try:
            dunder_r = R._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for R tensor")
        except AttributeError as e:
            raise Exception(f"R tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        # Test that np.array_equal returns True when comparing a Tensor with itself
        self.assertTrue(np.array_equal(self.test_tensor, self.test_tensor), msg="Tensor not correctly compared to itself")

        # Test that np.linalg.norm still returns a float as expected
        self.assertTrue(type(np.linalg.norm(self.test_tensor)) == np.float64, msg="Norm is not np.float64")
