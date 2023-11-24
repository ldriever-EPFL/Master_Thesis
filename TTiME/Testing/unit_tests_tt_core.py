import os
import sys
import unittest
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from ttime import Tensor, TTCore


class TestTTCore(unittest.TestCase):
    """
    Testing the TTCore class in tt_core.py

    Certain aspects have already been tested in the parent Tensor class and are thus not re-tested here
    """

    def setUp(self):
        np.random.seed(99)
        self.test_core = TTCore(np.random.randint(0, 100, (3, 10, 4)))

    def test_init(self):
        """
        Check that a TTCore cannot be initialized with an array of a dimension other than 3
        """
        with self.assertRaises(Exception) as context:
            _ = TTCore(np.random.randint(0, 100, (3, 10)))
        self.assertEqual(str(context.exception), 'A TTCore must be a third order tensor')

        with self.assertRaises(Exception) as context:
            _ = TTCore(np.random.randint(0, 100, (3, 10, 4, 2)))
        self.assertEqual(str(context.exception), 'A TTCore must be a third order tensor')

    def test_slicing_object_type(self):
        """
        Ensure that slicing returns a TTCore if the slice is of dimension 3, and returns a Tensor  instance otherwise
        """
        # Check indexed entry for correctness of slicing
        self.assertEqual(self.test_core[1, 1, 1], 53)

        # Check 2d slice
        self.assertTrue(isinstance(self.test_core[1], Tensor))

        # Check 3d slice
        self.assertTrue(isinstance(self.test_core[1:], TTCore))

    def test_ufunc(self):
        """
        Ensure that the np.ndarray ufuncs return TTCore instances if the output dimension is appropriate and Tensor instances otherwise.
        The correct functioning of the ufuncs follows from the testing of the Tensor class
        """
        # Test for square root
        sqrt_core = np.sqrt(self.test_core)
        self.assertEqual(TTCore, type(sqrt_core), msg="square rooted TTCore is not a member of the TTCore class")
        try:
            dunder_r = sqrt_core._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for square rooted TTCore")
        except AttributeError as e:
            raise Exception(f"Square rooted TTCore is not a proper instance of the TTCore class. Initial error message: {e}")

        # Test for sum. That changes the dimension and should thus return a Tensor, not a TTCore
        summed_tensor = self.test_core.sum(axis=1)
        self.assertEqual(Tensor, type(summed_tensor), msg="summed tensor is not a member of the Tensor class")
        self.assertEqual(454, summed_tensor[0, 2], msg='summed tensor does not have the expected values')
        try:
            dunder_r = summed_tensor._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for summed tensor Tensor")
        except AttributeError as e:
            raise Exception(f"summed tensor is not a proper instance of the Tensor class. Initial error message: {e}")

    def test_general_func(self):
        """
        There are a lot of handy functions defined in other parts of the numpy package, and many will return ndarray instances.
        We check that, when using these functions on Tensor instances, the result too is a Tensor instance
        """
        # Test that np.linalg.inv returns a TTCore
        inv = np.linalg.inv(self.test_core[:, :4])
        self.assertEqual(TTCore, type(inv), msg="inv is not a member of the Tensor class")
        try:
            dunder_r = inv._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for inv tensor")
        except AttributeError as e:
            raise Exception(f"inv tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        # Test that concatenating a TTCore to itself gives a TTCore of expected shape
        result = np.concatenate((self.test_core, self.test_core), axis=1)
        self.assertEqual(TTCore, type(result), msg="result is not a member of the TTCore class")
        np.testing.assert_array_equal((3, 20, 4), result.n)
        try:
            dunder_r = result._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for result tensor")
        except AttributeError as e:
            raise Exception(f"result tensor is not a proper instance of the TTCore class. Initial error message: {e}")

        # Test that np.linalg.qr returns two TTCore instances
        Q, R = np.linalg.qr(self.test_core)
        self.assertEqual(TTCore, type(Q), msg="Q tensor is not a member of the TTCore class")
        try:
            dunder_r = Q._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for Q TTCore")
        except AttributeError as e:
            raise Exception(f"Q tensor is not a proper instance of the TTCore class. Initial error message: {e}")

        self.assertEqual(TTCore, type(R), msg="R tensor is not a member of the TTCore class")
        try:
            dunder_r = R._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for R TTCore")
        except AttributeError as e:
            raise Exception(f"R tensor is not a proper instance of the TTCore class. Initial error message: {e}")

        # Test that summing along one dimension (i.e. reducing the core to 2d) returns a Tensor instance
        # Unlike in the ufunc test we now use np.sum() instead of .sum()
        summed_tensor = np.sum(self.test_core, axis=1)
        self.assertEqual(Tensor, type(summed_tensor), msg="summed tensor is not a member of the Tensor class")
        np.testing.assert_array_equal((3, 4), summed_tensor.n, err_msg='summed tensor does not have the expected shape')
        self.assertEqual(454, summed_tensor[0, 2], msg='summed tensor does not have the expected values')
        try:
            dunder_r = summed_tensor._Tensor__r
            self.assertTrue(dunder_r is None, msg="__r is not None for summed tensor Tensor")
        except AttributeError as e:
            raise Exception(f"summed tensor is not a proper instance of the Tensor class. Initial error message: {e}")

        # Test that np.array_equal returns True when comparing a TTCore with itself
        self.assertTrue(np.array_equal(self.test_core, self.test_core), msg="TTCore not correctly compared to itself")

        # Test that np.linalg.norm still returns a float as expected
        self.assertTrue(type(np.linalg.norm(self.test_core)) == np.float64, msg="Norm is not np.float64")

    def test_L(self):
        """
        The correctness of the unfolding function has been verified for the Tensor class.
        Thus, here we only check that the correct unfolding is applied, which we can do by checking the resulting matrix shape
        """
        np.testing.assert_array_equal((30, 4), self.test_core.L.shape, err_msg="Left unfolding has the wrong shape")

    def test_L_assignment(self):
        """
        It is possible to change the TTCore by assigning new values to the left unfolding, but only if the shape matches.
        Otherwise an error should be raised. Assignment of individual elements in the left unfolding is not supported
        """
        # Check that assignment is not allowed when the new array shape is different
        with self.assertRaises(Exception) as context:
            self.test_core.L = np.random.randint(0, 100, (3, 10))
        self.assertEqual(str(context.exception), "When assigning new values to the left unfolding using the = operator, the new value must be a matrix of equal shape to the left unfolding.\nIf the number of columns changed use obj = obj.from_L(matrix) instead")

        # Check that assignment works when changing the entire matrix
        test_matrix = np.random.randint(101, 200, (30, 4))  # Values are guaranteed to be different from original ones
        self.test_core.L = test_matrix
        # Test equivalence using three randomly selected points
        self.assertEqual(test_matrix[0, 0], self.test_core[0, 0, 0], "Test point value incorrect for L_assignment")
        self.assertEqual(test_matrix[16, 3], self.test_core[1, 5, 3], "Test point value incorrect for L_assignment")
        self.assertEqual(test_matrix[29, 2], self.test_core[2, 9, 2], "Test point value incorrect for L_assignment")

    def test_from_L(self):
        """
        The from_L function allows replacing the left unfolding with a matrix of different size, but still compatible shape
        """
        # Test that the matrix dimensions are checked
        with self.assertRaises(Exception) as context:
            _ = self.test_core.from_L(np.random.randint(0, 100, (30, 4, 2)))
        self.assertEqual(str(context.exception), "When getting a TTCore from its left unfolding, the new value has to be a matrix with an equal number of rows as the left unfolding")

        with self.assertRaises(Exception) as context:
            _ = self.test_core.from_L(np.random.randint(0, 100, (29, 4)))
        self.assertEqual(str(context.exception), "When getting a TTCore from its left unfolding, the new value has to be a matrix with an equal number of rows as the left unfolding")

        # Test that the function works as expected for a valid input
        test_matrix = np.random.randint(101, 200, (30, 2))  # Values are guaranteed to be different from original ones
        self.test_core = self.test_core.from_L(test_matrix)
        np.testing.assert_array_equal((3, 10, 2), self.test_core.n, 'Shape of new TTCore is not as expected')
        np.testing.assert_array_equal(test_matrix, self.test_core.L, 'Values of new TTCore are not as expected')  # If the left unfolding of the new TTCore equals the test matrix, all values must have been assigned to the correct spots

    def test_R(self):
        """
        The correctness of the unfolding function has been verified for the Tensor class.
        Thus, here we only check that the correct unfolding is applied, which we can do by checking the resulting matrix shape
        """
        np.testing.assert_array_equal((3, 40), self.test_core.R.shape, err_msg="Right unfolding has the wrong shape")

    def test_R_assignment(self):
        """
        It is possible to change the TTCore by assigning new values to the right unfolding, but only if the shape matches.
        Otherwise an error should be raised. Assignment of individual elements in the right unfolding is not supported
        """
        # Check that assignment is not allowed when the new array shape is different
        with self.assertRaises(Exception) as context:
            self.test_core.R = np.random.randint(0, 100, (3, 10))
        self.assertEqual(str(context.exception), "When assigning new values to the right unfolding using the = operator, the new value must be a matrix of equal shape to the right unfolding.\nIf the number of rows changed use obj = obj.from_R(matrix) instead")

        # Check that assignment works when changing the entire matrix
        test_matrix = np.random.randint(101, 200, (3, 40))  # Values are guaranteed to be different from original ones
        self.test_core.R = test_matrix
        # Test equivalence using three randomly selected points
        self.assertEqual(test_matrix[0, 0], self.test_core[0, 0, 0], "Test point value incorrect for R_assignment")
        self.assertEqual(test_matrix[2, 17], self.test_core[2, 7, 1], "Test point value incorrect for R_assignment")
        self.assertEqual(test_matrix[1, 34], self.test_core[1, 4, 3], "Test point value incorrect for R_assignment")

    def test_from_R(self):
        """
        The from_R function allows replacing the left unfolding with a matrix of different size, but still compatible shape
        """
        # Test that the matrix dimensions are checked
        with self.assertRaises(Exception) as context:
            _ = self.test_core.from_R(np.random.randint(0, 100, (3, 40, 2)))
        self.assertEqual(str(context.exception), "When getting a TTCore from its right unfolding, the new value has to be a matrix with an equal number of columns as the right unfolding")

        with self.assertRaises(Exception) as context:
            _ = self.test_core.from_R(np.random.randint(0, 100, (3, 39)))
        self.assertEqual(str(context.exception), "When getting a TTCore from its right unfolding, the new value has to be a matrix with an equal number of columns as the right unfolding")

        # Test that the function works as expected for a valid input
        test_matrix = np.random.randint(101, 200, (1, 40))  # Values are guaranteed to be different from original ones
        self.test_core = self.test_core.from_R(test_matrix)
        np.testing.assert_array_equal((1, 10, 4), self.test_core.n, 'Shape of new TTCore is not as expected')
        np.testing.assert_array_equal(test_matrix, self.test_core.R, 'Values of new TTCore are not as expected')  # If the right unfolding of the new TTCore equals the test matrix, all values must have been assigned to the correct spots
