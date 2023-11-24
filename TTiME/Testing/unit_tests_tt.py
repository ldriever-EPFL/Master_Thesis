import os
import sys
import unittest
import warnings
import numpy as np
from copy import deepcopy
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '../'))
from ttime import TT, TTCore


class TestTT(unittest.TestCase):
    """
    Testing the TT class in tt.py
    """

    def setUp(self):
        np.random.seed(99)
        self.test_core_list = [np.random.random((1, 13, 3)), np.random.random((3, 17, 2)), np.random.random((2, 9, 5)), np.random.random((5, 10, 4)), np.random.random((4, 11, 1))]
        self.test_tt = TT(self.test_core_list)

    def test_init(self):
        """
        Test that the input cores undergo the intended checks and that all are converted to TTCore instances
        """
        # Check that all cores are converted to TTCore instances
        for i in range(5):
            self.assertIsInstance(self.test_tt.cores[i], TTCore)

        # Check that non-3D cores are detected and rejected
        local_core_list = self.test_core_list[:]
        local_core_list.append(np.random.random((1, 13, 7, 1)))
        with self.assertRaises(Exception) as context:
            _ = TT(local_core_list)
        self.assertEqual(str(context.exception), "All input cores must be third order tensors")

        # Test that the first core must indeed have size 1 along dimension 0
        local_core_list = self.test_core_list[:]
        local_core_list[0] = np.random.random((2, 13, 3))
        with self.assertRaises(Exception) as context:
            _ = TT(local_core_list)
        self.assertEqual(str(context.exception), "The first core must have size 1 along dimension 0")

        # Test that the last core must indeed have size 1 along dimension 2
        local_core_list = self.test_core_list[:]
        local_core_list[4] = np.random.random((4, 11, 2))
        with self.assertRaises(Exception) as context:
            _ = TT(local_core_list)
        self.assertEqual(str(context.exception), "The last core must have size 1 along dimension 2")

        # Test that the cores are checked for compatible shapes
        local_core_list = self.test_core_list[:]
        local_core_list[1] = np.random.random((4, 11, 2))
        with self.assertRaises(Exception) as context:
            _ = TT(local_core_list)
        self.assertEqual(str(context.exception), "The cores at positions 0 and 1 have incompatible shapes")

    def test_geometric_properties(self):
        """
        Test the properties d, r, and n before and after changing the list of cores, thereby ensuring that they reflect the current state of the TT
        The rank array r is called a geometric property here, as it purely depends on the shapes of the cores
        """
        # First check the properties of the test TT
        self.assertEqual(5, self.test_tt.d, msg='TT has incorrect dimension')
        np.testing.assert_array_equal((1, 3, 2, 5, 4, 1), self.test_tt.r, err_msg='Incorrect rank array r')
        np.testing.assert_array_equal((13, 17, 9, 10, 11), self.test_tt.n, err_msg='Incorrect tensor shape')

        # Then add an additional core with incompatible dimensions. Check that calling TT.r raises the expected errors
        self.test_tt.cores.append(np.random.random((1, 13, 3)))
        with self.assertRaises(Exception) as context:
            _ = self.test_tt.r
        self.assertEqual(str(context.exception), 'The first and last cores must be such that r[0] and r[-1] equal 1')

        self.test_tt.cores[-1] = np.random.random((3, 13, 1))
        with self.assertRaises(Exception) as context:
            _ = self.test_tt.r
        self.assertEqual(str(context.exception), "The cores at positions 4 and 5 have incompatible shapes")

        # Then add a valid sixth core and re-test r, d, and n
        self.test_tt.cores[-1] = np.random.random((1, 13, 1))
        self.assertEqual(6, self.test_tt.d, msg='TT has incorrect dimension after adding additional core')
        np.testing.assert_array_equal((1, 3, 2, 5, 4, 1, 1), self.test_tt.r, err_msg='Incorrect rank array r after adding additional core')
        np.testing.assert_array_equal((13, 17, 9, 10, 11, 13), self.test_tt.n, err_msg='Incorrect tensor shape after adding additional core')

    def test_item_getter(self):
        """
        Compute the entry using the nested sum formulation (see Steinlechner 2016 {doi.org/10.1137/15M1010506} pg 4) as opposed to the direct matrix formulation implemented in the class
        """
        test_indices = [1, 10, 8, 2, 0]

        # Compute the expected value
        expected_value = 0
        for k1 in range(3):
            for k2 in range(2):
                for k3 in range(5):
                    for k4 in range(4):
                        expected_value += self.test_core_list[0][0, 1, k1] * self.test_core_list[1][k1, 10, k2] * self.test_core_list[2][k2, 8, k3] * self.test_core_list[3][k3, 2, k4] * self.test_core_list[4][k4, 0, 0]

        # Do the comparison
        self.assertAlmostEqual(expected_value, self.test_tt[test_indices], msg="Indexed values is incorrect")

    def test_full_tensor(self):
        """
        Test that the two methods for accessing the fully computed tensor have the intended size-checks in place and that the correct result is computed.
        Because the item getter method is already verified in a different test, we can use it for verifying the full tensor by checking randomly selected entries
        """
        # The default tensor is already larger than the safety limit of 1e5 entries. Check that it leads to the expected exceptions/warning
        with self.assertRaises(Exception) as context:
            _ = self.test_tt.full_tensor
        self.assertEqual(str(context.exception), "The full tensor has more than 1e5 entries. To nonetheless compute it, use obj.get_fulltensor(size_error=False)")

        with self.assertRaises(Exception) as context:
            _ = self.test_tt.get_full_tensor()
        self.assertEqual(str(context.exception), "The full tensor has more than 1e5 entries. To nonetheless compute it, set size_error=False")

        with warnings.catch_warnings(record=True) as issued_warning:
            full_tensor = self.test_tt.get_full_tensor(size_error=False)
        self.assertEqual("The full tensor has more than 1e5 entries", str(issued_warning[0].message))

        # Check that the tensor shape is correct
        np.testing.assert_array_equal(self.test_tt.n, full_tensor.shape, f"Full tensor has incorrect shape. Shape is {full_tensor.shape}")

        # Check using three randomly selected elements that the full tensor is correct
        rndm_point_1 = tuple(np.random.randint((0, 0, 0, 0, 0), self.test_tt.n, 5))
        rndm_point_2 = tuple(np.random.randint((0, 0, 0, 0, 0), self.test_tt.n, 5))
        rndm_point_3 = tuple(np.random.randint((0, 0, 0, 0, 0), self.test_tt.n, 5))

        self.assertEqual(self.test_tt[rndm_point_1], full_tensor[rndm_point_1], "Incorrect value for test point 1")
        self.assertEqual(self.test_tt[rndm_point_2], full_tensor[rndm_point_2], "Incorrect value for test point 2")
        self.assertEqual(self.test_tt[rndm_point_3], full_tensor[rndm_point_3], "Incorrect value for test point 3")

        # Check that a small tensor can be accessed using the 'full_tensor' property and that this changes when the cores are changed
        small_TT = TT([np.random.random((1, 13, 3)), np.random.random((3, 9, 5)), np.random.random((5, 10, 4)), np.random.random((4, 11, 1))])
        small_full_T1 = small_TT.full_tensor  # That this line does not throw an Exception (which would cause the test to fail) verifies that the 'full_tensor' property can be used here
        small_TT.cores[1][0, 0, 0] = 99  # entry is guaranteed to be out of range of original values
        self.assertFalse(np.array_equal(small_full_T1, small_TT.full_tensor), msg='Changing cores did not affect the full tensor')

    def test_norm(self):
        """
        We test the efficient method implemented for the .norm property against the inefficient method of computing the full tensor and taking its norm
        Assumes that obj.full_tensor is correct (which is verified in a seperate test)
        """
        # Use a smaller tensor to avoid unnecessarily high computational cost
        small_TT = TT([np.random.random((1, 13, 3)), np.random.random((3, 9, 5)), np.random.random((5, 10, 4)), np.random.random((4, 11, 1))])

        self.assertAlmostEqual(np.linalg.norm(small_TT.full_tensor), small_TT.norm, msg='Tensor norm is incorrect')

    def test_mu_orthogonalize(self):
        """
        As explained in Steinlechner 2016 {doi.org/10.1137/15M1010506}, a mu-orthogonal TT satisfies that U.L.T @ U.L gives the appropriately sized identity matrix
        for all cores U left of core mu, and that W.R @ W.R.T gives the appropriately sized identity matrix for all cores W to the right of core mu.

        In line with the notation of the paper, core mu is the one which, in Python, has index mu-1
        """
        # Check that only integer mu in the range [1, d] are allowed
        with self.assertRaises(Exception) as context:
            self.test_tt.mu_orthogonalize(2.0)
        self.assertEqual(str(context.exception), "mu must be an integer in the range [1, 5]")

        with self.assertRaises(Exception) as context:
            self.test_tt.mu_orthogonalize(0)
        self.assertEqual(str(context.exception), "mu must be an integer in the range [1, 5]")

        with self.assertRaises(Exception) as context:
            self.test_tt.mu_orthogonalize(6)
        self.assertEqual(str(context.exception), "mu must be an integer in the range [1, 5]")

        # Orthogonalize for an arbitrarily mu (here mu=2) and test that the orthogonality conditions are satisfied
        self.test_tt.mu_orthogonalize(2)
        np.testing.assert_array_almost_equal(np.eye(3), self.test_tt.cores[0].L.T @ self.test_tt.cores[0].L, err_msg='First core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(2), self.test_tt.cores[2].R @ self.test_tt.cores[2].R.T, err_msg='Third core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(5), self.test_tt.cores[3].R @ self.test_tt.cores[3].R.T, err_msg='Fourth core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(4), self.test_tt.cores[4].R @ self.test_tt.cores[4].R.T, err_msg='Fifth core does not satisfy the orthogonality condition')

    def test_left_right_orthogonalize(self):
        """
        A left orthogonal TT is one that is d-orthogonal, a right orthogonal is one that is 1-orthogonal
        We use the same testing method as in 'test_mu_orthogonalize'
        """
        self.test_tt.left_orthogonalize()
        for i in range(4):
            np.testing.assert_array_almost_equal(np.eye(self.test_tt.r[i + 1]), self.test_tt.cores[i].L.T @ self.test_tt.cores[i].L, err_msg=f'Core {i} breaks left-orthogonality')

        self.test_tt.right_orthogonalize()
        for i in range(1, 5):
            np.testing.assert_array_almost_equal(np.eye(self.test_tt.r[i]), self.test_tt.cores[i].R @ self.test_tt.cores[i].R.T, err_msg=f'Core {i} breaks right-orthogonality')

    def test_orthogonality_check(self):
        """
        The private __check_orthogonality() method keeps track of whether the current list of cores is known to be orthogonal.
        It should return True if the TT has been orthogonalized and the cores are identical to their state after orthogonalization.
        If the cores have not yet been orthogonalized or have been changed since orthogonalizing, we do not know if they are orthgoonal so False is returned
        """
        # Initially both __orthogonal_mu and __orthogonal_comparison_cores should be None and the check should return False
        self.assertEqual(None, self.test_tt._TT__orthogonal_mu, msg="initial __orthogonal_mu is not None")
        self.assertEqual(None, self.test_tt._TT__orthogonal_comparison_cores, msg="initial __orthogonal_comparison_cores is not None")
        self.assertFalse(self.test_tt._TT__check_orthogonality(), msg="Initial orthogonality check does not return False")

        # After any orthogonalization (here we 2-orthogonalize) the check should return True and the parameters should have the expected values
        self.test_tt.mu_orthogonalize(2)
        self.assertEqual(2, self.test_tt._TT__orthogonal_mu, msg="__orthogonal_mu does not indicate last known orthogonalization mu")
        for i in range(5):
            np.testing.assert_array_equal(self.test_tt.cores[i], self.test_tt._TT__orthogonal_comparison_cores[i], err_msg=f"__orthogonal_comparison_cores core {i} does not match TT.cores core {i} right after orthogonalization")
        self.assertTrue(self.test_tt._TT__check_orthogonality(), msg="Orthogonality check after orthogonalizing does not return True")

        # Now we change one of the cores (the choice is arbitrary). Before running __check_orthogonality(), __orthogonal_mu should still be 2 and __orthogonal_comparison_cores should be not None, but should be different from the current cores (testing this ensures that a proper copy is made of the cores)
        self.test_tt.cores[0][0, 0, 0] = 99  # Value is guaranteed to be outside of range of initial values
        self.assertEqual(2, self.test_tt._TT__orthogonal_mu, msg="__orthogonal_mu does not indicate last known orthogonalization mu")

        local_bool = True
        for i in range(5):
            local_bool = local_bool & np.array_equal(self.test_tt.cores[i], self.test_tt._TT__orthogonal_comparison_cores[i])
        self.assertFalse(local_bool, msg="__orthogonal_comparison_cores match TT.cores after cores have been changed, but before re-orthogonalization")

        # If we now run orthogonality __check_orthogonality() it should return False and both __orthogonal_mu and __orthogonal_comparison_cores should be None
        self.assertFalse(self.test_tt._TT__check_orthogonality(), msg="Orthogonality check does not return False after modifying cores")
        self.assertEqual(None, self.test_tt._TT__orthogonal_mu, msg="__orthogonal_mu is not None after running unsuccessful orthogonality check")

    def test_single_rank_truncation(self):
        """
        Test that truncating a single rank at a specified mu changes TT.r in the expected way, and that the resulting TT is indeed mu - 1 orthogonal.
        Also check that 'truncating' to the current rank does not change the tensor entries
        """
        # Test that the relevant Exceptions will be raised
        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank(2, 5)
        self.assertEqual(str(context.exception), "mu must be an integer in the range [1, 4]")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank(0, 2)
        self.assertEqual(str(context.exception), "new_r values cannot be zero, negative, or greater than the current ones")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank(6, 2)
        self.assertEqual(str(context.exception), "new_r values cannot be zero, negative, or greater than the current ones")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank(1., 2)
        self.assertEqual(str(context.exception), "new_r must be an integer or an iterable of integers of length 4")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank(1)
        self.assertEqual(str(context.exception),"When new_r is a single integer, mu must be an integer as well")

        # Truncate the rank at mu=3 from 5 to 3
        self.test_tt.truncate_rank(3, 3)

        # Check the new rank
        np.testing.assert_array_equal((1, 3, 2, 3, 4, 1), self.test_tt.r, err_msg="TT rank after truncation is not as expected")

        # Check that the TT is indeed 3-orthogonal and that the orthogonality result has been stored
        np.testing.assert_array_almost_equal(np.eye(3), self.test_tt.cores[0].L.T @ self.test_tt.cores[0].L, err_msg='First core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(2), self.test_tt.cores[1].L.T @ self.test_tt.cores[1].L, err_msg='Second core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(3), self.test_tt.cores[3].R @ self.test_tt.cores[3].R.T, err_msg='Fourth core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(4), self.test_tt.cores[4].R @ self.test_tt.cores[4].R.T, err_msg='Fifth core does not satisfy the orthogonality condition')

        self.assertEqual(3, self.test_tt._TT__orthogonal_mu, msg="__orthogonal_mu does not indicate last known orthogonalization mu")
        for i in range(5):
            np.testing.assert_array_equal(self.test_tt.cores[i], self.test_tt._TT__orthogonal_comparison_cores[i], err_msg=f"__orthogonal_comparison_cores core {i} does not match TT.cores core {i}")

        # Check that 'truncating' the rank at 4 from 4 to 4 does not change the tensor
        rand_idx = tuple(np.random.randint((0, 0, 0, 0, 0), self.test_tt.n, 5))
        ref_value = self.test_tt[rand_idx]
        self.test_tt.truncate_rank(4, 4)
        self.assertAlmostEqual(float(ref_value), float(self.test_tt[rand_idx]), msg="'Truncating' to current rank changed the tensor")  # Conversion to floats is necessary for compatibility with inbuilt rounding method

    def test_all_rank_truncation(self):
        """
        Test that rank truncation by specifying a new rank tuple changes TT.r in the expected way, and that the resulting TT is indeed 1 orthogonal.
        """
        # Test that the relevant Exceptions will be raised
        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank((1, 3, 2, 3, 4, 1))
        self.assertEqual(str(context.exception), "new_r must be an integer or an iterable of integers of length 4 (do not include the leading and trailing 1s of the rank tuple))")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank((3, 2, 3, 4.))
        self.assertEqual(str(context.exception), "all elements of new_r must be integers")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank((3, 0, 3, 4))
        self.assertEqual(str(context.exception), "new rank values cannot be zero, negative, or greater than the current ones")

        with self.assertRaises(Exception) as context:
            self.test_tt.truncate_rank((4, 2, 3, 4))
        self.assertEqual(str(context.exception), "new rank values cannot be zero, negative, or greater than the current ones")

        # Check that specifying a mu value will give a warning
        with warnings.catch_warnings(record=True) as issued_warning:
            self.test_tt.truncate_rank((3, 2, 3, 4), 2)
        self.assertEqual('setting mu when new_r is an iterable has no effect', str(issued_warning[0].message))

        # Re-create the original TT and truncate its ranks using the proper procedure
        self.test_tt = TT(self.test_core_list)
        self.test_tt.truncate_rank((2, 1, 3, 2))

        # Check the new rank
        np.testing.assert_array_equal((1, 2, 1, 3, 2, 1), self.test_tt.r, err_msg="TT rank after truncation is not as expected")

        # Check that the TT is indeed 1-orthogonal and that the orthogonality result has been stored
        np.testing.assert_array_almost_equal(np.eye(2), self.test_tt.cores[1].R @ self.test_tt.cores[1].R.T, err_msg='Second core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(1), self.test_tt.cores[2].R @ self.test_tt.cores[2].R.T, err_msg='Third core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(3), self.test_tt.cores[3].R @ self.test_tt.cores[3].R.T, err_msg='Fourth core does not satisfy the orthogonality condition')
        np.testing.assert_array_almost_equal(np.eye(2), self.test_tt.cores[4].R @ self.test_tt.cores[4].R.T, err_msg='Fifth core does not satisfy the orthogonality condition')

        self.assertEqual(1, self.test_tt._TT__orthogonal_mu, msg="__orthogonal_mu does not indicate last known orthogonalization mu")
        for i in range(5):
            np.testing.assert_array_equal(self.test_tt.cores[i], self.test_tt._TT__orthogonal_comparison_cores[i], err_msg=f"__orthogonal_comparison_cores core {i} does not match TT.cores core {i}")

    def test_random_vec_rank_increase(self):
        """
        Test that using the random_vec_rank_increase function indeed increases the desired rank and keeps the core shapes compatible
        """
        # Test that the relevant Exceptions will be raised
        with self.assertRaises(Exception) as context:
            self.test_tt.random_vec_rank_increase(0)
        self.assertEqual(str(context.exception), "mu must be an integer in the range [1, 4]")

        # Perform the rank increase
        self.test_tt.random_vec_rank_increase(2)
        # As verified in a different test, calling TT.r will automatically detect any shape mismatches between the cores
        np.testing.assert_array_equal((1, 3, 3, 5, 4, 1), self.test_tt.r, err_msg="TT rank did not increase as expected")

    def test_mode_mu_multiply(self):
        """
        Test that using the mode_mu_multiply function indeed only changes the core with index mu-1 and changes it in the expected way
        """

        og_cores = deepcopy(self.test_tt.cores)
        wrong_A1 = np.zeros((10, 7))
        wrong_A2 = np.zeros((1, 1, 1))
        A = np.random.random((7, 10))
        expected_core = np.zeros((5, 7, 4))
        for i in range(5):
            for j in range(7):
                for k in range(4):
                    expected_core[i, j, k] = np.dot(og_cores[3][i, :, k], A[j, :])

        # Test that the expected Exceptions are raised
        with self.assertRaises(Exception) as context:
            self.test_tt.mode_mu_multiply(wrong_A1, 1.2)
        self.assertEqual(str(context.exception), "mu must be an integer in the range [1, 5]")

        with self.assertRaises(Exception) as context:
            self.test_tt.mode_mu_multiply(wrong_A1, 3)
        self.assertEqual(str(context.exception), "Dimension 1 of A must match dimension 1 of the TT core with index mu-1")

        with self.assertRaises(Exception) as context:
            self.test_tt.mode_mu_multiply(wrong_A2, 3)
        self.assertEqual(str(context.exception), "A must be a 2D Tensor object or numpy array")

        # Test that the multiplication result is as expected and that the other cores remain unaffected
        self.test_tt.mode_mu_multiply(A, 4)
        np.testing.assert_array_equal((5, 7, 4), self.test_tt.cores[3].shape, err_msg='Multiplied core does not have the expected shape')
        np.testing.assert_array_almost_equal(expected_core, self.test_tt.cores[3], err_msg='Multiplied core is not as expected')

        for i in [0, 1, 2, 4]:
            np.testing.assert_array_equal(og_cores[i], self.test_tt.cores[i], err_msg=f'The core with index {i} unexpectedly changed')
