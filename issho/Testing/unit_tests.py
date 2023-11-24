import unittest
import os
from os.path import join
import sys
import shutil
import numpy as np
base_path = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the Testing directory, ensuring that all testing files are found and saved to the correct location
sys.path.append(join(base_path, '../'))  # Absolute path to the Issho_Package directory
import issho
from issho import Airfoil, FoilBlock, FoilChain, FixedFoilChain


class TestAirfoil(unittest.TestCase):
    """
    Testing the Airfoil class in airfoil.py
    """
    def setUp(self):
        self.airfoil1 = Airfoil('unit_test_profile', [-0.1, 0], [0.9, -0.1], [0.1, 0], 1, 1.5, 0.8, base_path=base_path)
        self.airfoil2 = Airfoil('my_airfoil', [-0.1, 0.1], None, [0.05, 0], 0.5, 0.7, 0.4, base_path=base_path)
        self.airfoil3 = Airfoil('my_airfoil', [-0.1, 0.1], None, [0.05, 0], 0.5, 0.7, profile=np.array([[1, 2], [0, 0], [1, 2]]), base_path=base_path)

    def test_parameter_reading(self):
        print(join(os.getcwd(), os.path.relpath(os.getcwd(), base_path), 'Test_FSS/results/'))

        self.assertEqual(self.airfoil1.name, 'unit_test_profile')
        self.assertEqual(self.airfoil1.pos_lh.tolist(), [-0.1, 0])
        self.assertEqual(self.airfoil1.pos_rh.tolist(), [0.9, -0.1])
        self.assertEqual(self.airfoil1.pos_cm.tolist(), [0.1, 0])
        self.assertEqual(self.airfoil1.m, 1)
        self.assertEqual(self.airfoil1.J, 1.5)
        self.assertEqual(self.airfoil1.l_chord, 0.8)
        self.assertEqual(self.airfoil3.profile.tolist(), [[1, 2], [0, 0], [1, 2]])

    def test_get_vector_angle(self):
        # Test the get_vector-angle function from helper.py. Included here, as this is its lowest-level use
        self.assertAlmostEqual(issho.helper.get_vector_angle([np.sqrt(3) / 2, 0.5]), np.pi / 6)

    def test_vector_computations(self):
        np.testing.assert_array_equal(self.airfoil1.hinge_vector, [1, -0.1])
        np.testing.assert_array_equal(self.airfoil2.hinge_vector, [0.5, -0.1])

        np.testing.assert_array_almost_equal(self.airfoil1.hinge_cm_vector, [0.2, 0])
        np.testing.assert_array_almost_equal(self.airfoil2.hinge_cm_vector, [0.15, -0.1])

        self.assertAlmostEqual(self.airfoil1.l_h, 1.004987562, 9)
        self.assertAlmostEqual(self.airfoil2.l_h, 0.509901951, 4)

        self.assertAlmostEqual(self.airfoil1.l_cm, 0.2, 9)
        self.assertAlmostEqual(self.airfoil2.l_cm, 0.180277564, 9)

        self.assertAlmostEqual(self.airfoil1.theta_h, -0.099668652, 9)
        self.assertAlmostEqual(self.airfoil2.theta_h, -0.19739556, 9)

        self.assertAlmostEqual(self.airfoil1.theta_cm, 0.099668652, 9)
        self.assertAlmostEqual(self.airfoil2.theta_cm, -0.588002604 + 0.19739556 , 9)

    def test_profile_reading(self):
        np.testing.assert_array_equal(self.airfoil1.profile, np.array([[1, 0], [0.5, 0.5], [0, 0], [0.5, -0.5], [1, 0]]) * 0.8)

    def test_profile_creation(self):
        np.testing.assert_array_equal(self.airfoil2.profile, np.array([[0.4, 0], [0, 0], [0.4, 0]]))


class TestFoilBlock(unittest.TestCase):
    """
    Testing the FoilBlock class in foil_block.py
    """
    def setUp(self):
        airfoil1 = Airfoil('unit_test_profile', [-0.1, 0], [0.9, -0.1], [0.1, 0], 1, 1.5, 0.8, base_path=base_path)
        airfoil2 = Airfoil('naca0012', [-0.1, 0.1], [0.5, 0], [0.05, 0.1], 0.5, 0.7, 0.4, base_path=base_path)
        airfoil3 = Airfoil('naca0012', [0.1, 0], None, [0.05, -0.10], 0.3, 0.4, 0.4, base_path=base_path)
        self.fb_angles = [0.3, -0.8]

        self.foil_block = FoilBlock([airfoil1, airfoil2, airfoil3], self.fb_angles)

    def test_basic_steps(self):
        self.assertEqual(self.foil_block.n, 3)
        np.testing.assert_array_equal(self.foil_block.component_angles, self.fb_angles)

    def test_rotate_function(self):
        # Test of rotate helper function is included here, as this is its lowest-level use
        # Test that an improper vector shape will raise the expected error:
        with self.assertRaises(Exception) as context:
            issho.helper.rotate(np.array([[1, 2], [3, 4], [5, 6]]), 0)

        self.assertEqual(str(context.exception), "Input vector must be of shape 2xn")

        # Test that the computation gives the expected result for a two-element vector
        vec = np.array([1, 1])
        angle = -np.pi / 4
        expected_vec = np.array([[np.sqrt(2)], [0]])

        np.testing.assert_array_almost_equal(issho.helper.rotate(vec, angle), expected_vec)

        # Test that the computation gives the expected result for a multi-element vector
        vec = np.array([[1, 0, 3], [0, 2, 4]])
        angle = np.pi / 3
        expected_vec = np.array([[0.5, -np.sqrt(3), 1.5 - 2 * np.sqrt(3)], [np.sqrt(3) / 2, 1, 3 / 2 * np.sqrt(3) + 2]])

        np.testing.assert_array_almost_equal(issho.helper.rotate(vec, angle), expected_vec)

        # Test that the computation gives the expected result for a multi-element vector with multiple angles
        vec = np.array([[1, 0, 3], [0, 2, 4]])
        angles = np.array([np.pi / 3, 0, -np.pi / 2])
        expected_vec = np.array([[0.5, 0, 4], [np.sqrt(3) / 2, 2, -3]])

        np.testing.assert_array_almost_equal(issho.helper.rotate(vec, angles), expected_vec)

    def test_vector_computations(self):
        """
        Test based on hand-calculations for the values specified above
        """
        self.assertAlmostEqual(self.foil_block.m, 1.8, 9, msg='Mismatch in total mass')
        np.testing.assert_array_almost_equal(self.foil_block.hinge_vector, [1.86602868271, -0.162049186497], err_msg='Mismatch in hinge vector')
        np.testing.assert_array_almost_equal(self.foil_block.hinge_cm_vector, [0.68051661, -0.02913252], err_msg='Mismatch in hinge cm vector')
        self.assertAlmostEqual(self.foil_block.l_h, 1.873051783464824, 9, msg='Mismatch in hinge vector length')
        self.assertAlmostEqual(self.foil_block.l_cm, 0.6811399022004884, 9, msg='Mismatch in hinge cm vector length')
        self.assertAlmostEqual(self.foil_block.theta_h, -0.08662442189114196, 9, msg='Mismatch in hinge vector angle')
        self.assertAlmostEqual(self.foil_block.theta_cm, 0.043841127179518616, 9, msg='Mismatch in hinge cm vector angle')
        self.assertAlmostEqual(self.foil_block.first_hinge_angle_diff, 0.01304423060002008, 9, msg='Mismatch in first hinge angle diff')
        self.assertAlmostEqual(self.foil_block.last_hinge_angle_diff, 0.41337557810885805, 9, msg='Mismatch in last hinge angle diff')
        self.assertAlmostEqual(self.foil_block.J, 3.146897331900336, 9, msg='Mismatch in total moment of inertia')


class TestFoilChain(unittest.TestCase):
    """
    Testing the FoilChain class in foil_chain.py
    """

    def setUp(self):
        self.foil_chain = None

    def test_init(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'general_FC_test.txt'), base_path=base_path)

        # Where applicable, calculations are done by hand or using the already verified Airfoil and FoilBlock classes

        self.assertEqual(self.foil_chain.n_airfoils, 5, msg='Incorrect number of airfoils')
        self.assertEqual(self.foil_chain.g, 1.2, msg='Incorrect gravity')
        self.assertEqual(self.foil_chain.theta_g, -0.03, msg='Incorrect gravity angle')
        np.testing.assert_equal(self.foil_chain.foil_block_indices_per_block, [[0], [1, 2, 3], [4]])
        self.assertEqual(self.foil_chain.n_foilblocks, 3, msg='Incorrect number of FoilBlocks')
        np.testing.assert_array_almost_equal(self.foil_chain.theta_x, np.cumsum([0.05, 0.1 + 0.35178216988370437, 0.3 + 0.16996838451293786]), err_msg='Incorrect theta_x')
        np.testing.assert_array_equal(self.foil_chain.theta_x_dot, [0, 0, 0], err_msg='Incorrect theta_x_dot')
        np.testing.assert_array_almost_equal(self.foil_chain.all_l_h, [0.7, np.sqrt(0.1), 0.25, 0.25, 0.25000199999200007], err_msg='Incorrect l_h')
        np.testing.assert_array_equal(self.foil_chain.airfoil_left_hinge_vectors, [[0.2, 0.0], [-0.05, 0.0], [-0.05, 0.0], [-0.05, 0.0], [-0.05, 0.0]], err_msg='Incorrect airfoil left hinge vectors')
        np.testing.assert_array_almost_equal(self.foil_chain.l_h, [0.7, 0.7842043763152309, 0.25000199999200007], err_msg='Incorrect l_h')
        np.testing.assert_array_almost_equal(self.foil_chain.l_cm, [0.09999999999999998, 0.3227307303600904, 0.1063014581273465], err_msg='Incorrect l_cm')
        np.testing.assert_array_almost_equal(self.foil_chain.theta_cm, [0.0, -0.06440401676738954, -0.722829978288496], err_msg='Incorrect theta_cm')
        np.testing.assert_array_equal(self.foil_chain.m, [1.2, 0.7, 0.2], err_msg='Incorrect m')
        np.testing.assert_array_almost_equal(self.foil_chain.J, [3.5, 3.0326310076110508, 0.9], err_msg='Incorrect J')
        self.assertEqual(self.foil_chain.profiles_x[3017], 0.01600000 * 0.2, msg="Incorrect x-profile array")  # Test correctness using randomly selected indices
        self.assertEqual(self.foil_chain.profiles_x[-4], 0.1, msg="Incorrect x-profile array")
        self.assertEqual(self.foil_chain.profiles_y[178], 0.02272095 * 0.8, msg="Incorrect y-profile array")  # Test correctness using randomly selected indices
        self.assertEqual(self.foil_chain.profiles_y[-2], 0, msg="Incorrect y-profile array")
        np.testing.assert_array_equal(self.foil_chain.foil_block_indices_per_airfoil, [0, 1, 1, 1, 2])
        np.testing.assert_array_equal(self.foil_chain.foil_block_first_airfoil_indices, [0, 1, 1, 1, 4])
        np.testing.assert_array_equal(self.foil_chain.profile_indices, [0, 2001, 4002, 4005, 4008, 4011], err_msg="Incorrect profile indices list")
        np.testing.assert_array_almost_equal(self.foil_chain.additional_rotation_vector_hinge, [0, -0.030031615487062114 - 0.32175055439664224, -0.030031615487062114 - 0.32175055439664224 + 0.3, -0.030031615487062114 - 0.32175055439664224 + 0.2, 0], err_msg='Incorrect additional hinge rotation vector')
        np.testing.assert_array_almost_equal(self.foil_chain.additional_rotation_vector_chord, [0, -0.030031615487062114, -0.030031615487062114 - 0.32175055439664224 + 0.3, -0.030031615487062114 - 0.32175055439664224 + 0.2, -0.003999978666871465], err_msg='Incorrect additional chord rotation vector')

    def test_init_moment_computation(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'FC_moment_computation.txt'), base_path=base_path)
        np.testing.assert_array_equal(self.foil_chain.moment_computation_matrix, [[0.01, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -0.005, 0.005, 0], [0, 0, -0.02, 0.02, 0, 0, -0.1, 0.1]], err_msg='Incorrect moment_computation_matrix')
        np.testing.assert_array_equal(self.foil_chain.spring_moment_vector, [-0.01 * 0.03, 0, 0, -0.02 * 0.04], err_msg='Incorrect spring_moment_vector')

    def test_init_fixed_airfoil(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'init_test_fixed_airfoils.txt'), base_path=base_path)
        self.assertEqual(self.foil_chain.n_airfoils, 4)
        np.testing.assert_array_almost_equal(self.foil_chain.theta_x, [0.14, 0.24, 0.14, 0.44], 10)

    def test_sim_matrix_creation(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'general_FC_test.txt'), base_path=base_path)

        # Values based on hand calculations and the parts verified in the init function
        a1 = - 1 / 1.2
        b1 = 1 / 1.2
        c1 = 0.1 * np.sin(0.05)
        d1 = 1 / 1.2
        e1 = - (1 / 1.2 + 1 / 0.7)
        f1 = 1 / 0.7
        g1 = - 0.1 * np.sin(0.05) + 0.7 * np.sin(0.05)
        h1 = 0.32273073 * np.sin(0.43737815)
        i1 = 1 / 0.7
        j1 = - (1 / 0.7 + 1 / 0.2)
        k1 = - 0.32273073 * np.sin(0.43737815) + 0.78420438 * np.sin(0.50178217)
        l1 = 0.10630146 * np.sin(0.24892058)

        a2 = a1
        b2 = b1
        c2 = - 0.1 * np.cos(0.05)
        d2 = d1
        e2 = e1
        f2 = f1
        g2 = - 0.7 * np.cos(0.05) + 0.1 * np.cos(0.05)
        h2 = - 0.32273073 * np.cos(0.43737815)
        i2 = i1
        j2 = j1
        k2 = -0.78420438 * np.cos(0.50178217) + 0.32273073 * np.cos(0.43737815)
        l2 = - 0.10630146 * np.cos(0.24892058)

        a3 = - 0.1 * np.sin(0.05)
        b3 = 0.1 * np.sin(0.05) - 0.7 * np.sin(0.05)
        c3 = 0.1 * np.cos(0.05)
        d3 = - 0.1 * np.cos(0.05) + 0.7 * np.cos(0.05)
        e3 = -3.5
        f3 = - 0.32273073 * np.sin(0.43737815)
        g3 = 0.32273073 * np.sin(0.43737815) - 0.78420438 * np.sin(0.50178217)
        h3 = 0.32273073 * np.cos(0.43737815)
        i3 = - 0.32273073 * np.cos(0.43737815) + 0.78420438 * np.cos(0.50178217)
        j3 = -3.03263101
        k3 = - 0.10630146 * np.sin(0.24892058)
        l3 = 0.10630146 * np.cos(0.24892058)
        m3 = -0.9

        expected_matrix = np.array([[a1, b1,  0,  0,  0,  0, c1,  0,  0],
                                    [d1, e1, f1,  0,  0,  0, g1, h1,  0],
                                    [ 0, i1, j1,  0,  0,  0,  0, k1, l1],
                                    [ 0,  0,  0, a2, b2,  0, c2,  0,  0],
                                    [ 0,  0,  0, d2, e2, f2, g2, h2,  0],
                                    [ 0,  0,  0,  0, i2, j2,  0, k2, l2],
                                    [a3, b3,  0, c3, d3,  0, e3,  0,  0],
                                    [ 0, f3, g3,  0, h3, i3,  0, j3,  0],
                                    [ 0,  0, k3,  0,  0, l3,  0,  0, m3]])

        np.testing.assert_array_almost_equal(expected_matrix, self.foil_chain._FoilChain__create_simulation_matrix(), err_msg='Sim matrix not constructed correctly')

    def test_forcing_vector_creation(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'general_FC_test.txt'), base_path=base_path)

        # Values based on hand calculations and the parts verified in the init function
        np.random.seed(99)
        F_x = np.random.random(3)
        F_y = np.random.random(3)
        M = np.random.random(3)

        def expected_vector(theta_x, theta_x_dot, F_x, F_y, M):
            F_x_loc = F_x + 1.2 * self.foil_chain.m * np.sin(-0.03)
            F_y_loc = F_y - 1.2 * self.foil_chain.m * np.cos(-0.03)
            M_loc = M + 1.2 * self.foil_chain.m * self.foil_chain.l_cm * np.cos(theta_x + self.foil_chain.theta_cm + 0.03)

            M_h = np.ravel(self.foil_chain.moment_computation_matrix @ np.hstack((theta_x, theta_x_dot)) + self.foil_chain.spring_moment_vector)
            expected_vector = np.array([-F_x_loc[0] / 1.2 - theta_x_dot[0] ** 2 * 0.1 * np.cos(theta_x[0] + 0),
                                        F_x_loc[0] / 1.2 - F_x_loc[1] / 0.7 + theta_x_dot[0] ** 2 * (0.1 * np.cos(theta_x[0] + 0) - 0.7 * np.cos(theta_x[0])) - theta_x_dot[1] ** 2 * 0.32273073 * np.cos(theta_x[1] + -0.06440402),
                                        F_x_loc[1] / 0.7 - F_x_loc[2] / 0.2 + theta_x_dot[1] ** 2 * (0.32273073 * np.cos(theta_x[1] + -0.06440402) - 0.78420438 * np.cos(theta_x[1])) - theta_x_dot[2] ** 2 * 0.10630146 * np.cos(theta_x[2] + -0.72282998),
                                        -F_y_loc[0] / 1.2 - theta_x_dot[0] ** 2 * 0.1 * np.sin(theta_x[0] + 0),
                                        F_y_loc[0] / 1.2 - F_y_loc[1] / 0.7 + theta_x_dot[0] ** 2 * (0.1 * np.sin(theta_x[0] + 0) - 0.7 * np.sin(theta_x[0])) - theta_x_dot[1] ** 2 * 0.32273073 * np.sin(theta_x[1] + -0.06440402),
                                        F_y_loc[1] / 0.7 - F_y_loc[2] / 0.2 + theta_x_dot[1] ** 2 * (0.32273073 * np.sin(theta_x[1] + -0.06440402) - 0.78420438 * np.sin(theta_x[1])) - theta_x_dot[2] ** 2 * 0.10630146 * np.sin(theta_x[2] + -0.72282998),
                                        M_loc[0] + M_h[0] - M_h[1] + 0.1 * (-F_x_loc[0] * np.sin(theta_x[0] + 0) + F_y_loc[0] * np.cos(theta_x[0] + 0)),
                                        M_loc[1] + M_h[1] - M_h[2] + 0.32273073 * (-F_x_loc[1] * np.sin(theta_x[1] + -0.06440402) + F_y_loc[1] * np.cos(theta_x[1] + -0.06440402)),
                                        M_loc[2] + M_h[2] + 0.10630146 * (-F_x_loc[2] * np.sin(theta_x[2] + -0.72282998) + F_y_loc[2] * np.cos(theta_x[2] + -0.72282998))])
            return expected_vector

        # in the first case the angular velocities should be zero
        theta_x = np.array([0.05, 0.50178217, 0.97175055])
        theta_x_dot = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(expected_vector(theta_x, theta_x_dot, F_x, F_y, M), self.foil_chain._FoilChain__create_forcing_vector(F_x, F_y, M), err_msg='Forcing vector not constructed correctly')

        # Now change theta_x and theta_x_dot and test again
        self.foil_chain.theta_x = theta_x = np.random.random(3)
        self.foil_chain.theta_x_dot = theta_x_dot = np.random.random(3)
        np.testing.assert_array_almost_equal(expected_vector(theta_x, theta_x_dot, F_x, F_y, M), self.foil_chain._FoilChain__create_forcing_vector(F_x, F_y, M), err_msg='Forcing vector not constructed correctly')

    def test_gravity_implementation(self):
        # In part 1 gravity and the cm vector are aligned, so there should not be any accleration
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'gravity_implementation_part_1.txt'), base_path=base_path)
        theta, theta_dot = self.foil_chain.simulate(0.01, 1, [0, 0, 0], initial_theta_x=0, initial_theta_x_dot=0, document_motion=True)

        self.assertAlmostEqual(theta[0, 0], 0)
        self.assertAlmostEqual(theta_dot[0, 0], 0)

        # In part 2 gravity and the cm vector are not aligned, and the expected acceleration is in the negative direction
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'gravity_implementation_part_2.txt'), base_path=base_path)
        _, theta_dot = self.foil_chain.simulate(0.01, 1, [0, 0, 0], initial_theta_x=0, initial_theta_x_dot=0, document_motion=True)

        self.assertLess(theta_dot[0, 0], 0)

    def test_simulation_spring_1(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'spring_test_1.txt'), base_path=base_path)

        # Test use of spring coefficient using one airfoil and known analytical solution
        # Uses parameters m = 1.2, J = 3.9, L_cm = 0.4, theta_rest = 0.3, theta_0 = 0, K = 0.15
        T = 2  # Simulate two seconds
        n_t = 1000  # use 1000 steps
        t = (np.arange(n_t) + 1) / n_t * T

        theoretical_pos = 0.3 * (1 - np.cos(np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * t))
        sim_theta_pos, _ = self.foil_chain.simulate(T, n_t, [0, 0, 0], initial_theta_x=0, initial_theta_x_dot=0, document_motion=True)

        np.testing.assert_array_almost_equal(theoretical_pos, sim_theta_pos.flatten())

    def test_simulation_spring_2(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'spring_test_2.txt'), base_path=base_path)

        # Test that the correct relative angle is used for two connected airfoils
        # Checks that the force is in the expected direction by confirming that airfoil 1 accelerates clockwise, airfoil 2 accelerates counterclockwise

        _, sim_theta_dot = self.foil_chain.simulate(0.1, 1, [[0, 0], [0, 0], [0, 0]], document_motion=True)

        self.assertTrue(sim_theta_dot[0] > 0)
        self.assertTrue(sim_theta_dot[1] < 0)

    def test_simulation_damped_1(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'damping_test_1.txt'), base_path=base_path)

        # Test use of damping coefficient using one airfoil and known analytical solution
        # Uses parameters m = 1.2, J = 3.9, L_cm = 0.4, theta_0 = 0, theta_dot_0 = 1.3, c = 0.15
        T = 1  # Simulate one second
        n_t = 1000  # use 1000 steps
        t = (np.arange(n_t) + 1) / n_t * T

        theoretical_theta_dot = 1.3 * np.exp(-t * 0.15 / (3.9 + 0.4 ** 2 * 1.2))
        _, sim_theta_dot = self.foil_chain.simulate(T, n_t, [0, 0, 0], initial_theta_x=0, initial_theta_x_dot=1.3, document_motion=True)

        np.testing.assert_array_almost_equal(theoretical_theta_dot, sim_theta_dot.flatten())

    def test_simulation_damped_2(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'damping_test_2.txt'), base_path=base_path)

        # Test that the correct relative angular velocity is used for two connected airfoils
        # Checks that the force is in the expected direction by confirming that airfoil 1 decelerates, airfoil 2 accelerates

        _, sim_theta_dot = self.foil_chain.simulate(0.1, 1, [[0, 0], [0, 0], [0, 0]], initial_theta_x_dot=[2, 1], document_motion=True)

        self.assertTrue(sim_theta_dot[0] < 2)
        self.assertTrue(sim_theta_dot[1] > 1)

    def test_simulation_spring_damped(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'spring_damped_test.txt'), base_path=base_path)

        # Test use of spring constant and damping coefficient using one airfoil and known analytical solution
        # Uses parameters m = 0.2, J = 0.9, L_cm = 0.15, theta_0 = 0, theta_dot_0 = 1.3, k = 0.02, theta_rest = 0.04, c = 0.1
        T = 1  # Simulate one second
        n_t = 1000  # use 1000 steps
        t = (np.arange(n_t) + 1) / n_t * T

        J_eqv = 0.9 + 0.2 * 0.15 ** 2
        K = 0.02
        C = 0.1
        complex_exponent = (-C + np.emath.sqrt(C ** 2 - 4 * J_eqv * K)) / (2 * J_eqv)
        a = np.real(complex_exponent)
        b = np.imag(complex_exponent)

        theoretical_theta = 0.04 + np.real(np.exp(a * t) * (-0.04 * np.cos(b * t) + (1.3 + 0.04 * a) / b * np.sin(b * t)))
        theoretical_theta_dot = a * np.real(np.exp(a * t) * (-0.04 * np.cos(b * t) + (1.3 + 0.04 * a) / b * np.sin(b * t))) + np.real(np.exp(a * t) * (0.04 * b * np.sin(b * t) + (1.3 + 0.04 * a) * np.cos(b * t)))
        sim_theta, sim_theta_dot = self.foil_chain.simulate(T, n_t, [0, 0, 0], initial_theta_x=0, initial_theta_x_dot=1.3, document_motion=True)

        np.testing.assert_array_almost_equal(theoretical_theta, sim_theta.flatten())
        np.testing.assert_array_almost_equal(theoretical_theta_dot, sim_theta_dot.flatten())

    def test_simulation_pin(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'pin_test.txt'), base_path=base_path)

        # Test use of pin hinge using one airfoil and known analytical solution
        # Uses parameters m = 1.2, J = 3.9, L_cm = 0.4, theta_0 = 0
        T = 1  # Simulate one second
        n_t = 1000  # use 1000 steps
        t = (np.arange(n_t) + 1) / n_t * T

        theoretical_theta = -0.5 * (0.7 / (3.9 + 0.4 ** 2 * 1.2)) * t ** 2
        sim_theta, _ = self.foil_chain.simulate(T, n_t, [0, 0, 0.7], document_motion=True)

        np.testing.assert_array_almost_equal(theoretical_theta, sim_theta.flatten())

    def test_simulation_theta_cm(self):
        # To test whether theta_cm is being considered correctly, we consider two airfoils with horizontal hinge lines connected with pin hinges, where the first airfoil is effectively massless
        # There will be two test cases, and in both a vertical force (positive y-direction) is applied at the hinge connecting the airfoils
        # In case 1 the cm of the second airfoil is along the hinge line, so a notable negative deflection is expected in the first instance
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'theta_cm_test_file_1.txt'), base_path=base_path)
        theta_storage, _ = self.foil_chain.simulate(0.1, 1, [[0, 0], [0, 1], [0, 0]], document_motion=True)
        self.assertTrue(theta_storage[1, -1] < 0)
        self.assertFalse(np.isclose(theta_storage[1, -1], 0, 4))

        # In case 2 the cm of the second airfoil is perpendicular to the hinge line, vertically above the hinge. Thus, zero deflection is expected in the first instance
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'theta_cm_test_file_2.txt'), base_path=base_path)
        theta_storage, _ = self.foil_chain.simulate(0.1, 1, [[0, 0], [0, 1], [0, 0]], document_motion=True)
        self.assertAlmostEqual(theta_storage[1, -1], 0, 4)

        # NOTE: both tests considered only the first time step, because as soon as the first airfoil rotates, it accelerates the hinge in a non-vertical direction and the motion becomes more complex

    def test_simulation_convergence(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'spring_test_1.txt'), base_path=base_path)

        # Test convergence behavior using one airfoil and known analytical solution on a non-linear problem
        # Uses parameters m = 1.2, J = 3.9, L_cm = 0.4, theta_0 = 0
        T = 10
        n_t = 10 ** 4
        t = (np.arange(n_t) + 1) / n_t * T

        initial_theta_x = -0.05
        initial_theta_x_dot = 0.1

        theoretical_theta = 0.3 + (initial_theta_x - 0.3) * np.cos(np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * t) + initial_theta_x_dot / np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * np.sin(np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * t)
        theoretical_theta_dot = -(initial_theta_x - 0.3) * np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * np.sin(np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * t) + initial_theta_x_dot * np.cos(np.sqrt(0.15 / (3.9 + 0.4 ** 2 * 1.2)) * t)
        sim_theta1, sim_theta_dot1 = self.foil_chain.simulate(T, 10, [0, 0, 0], initial_theta_x=initial_theta_x, initial_theta_x_dot=initial_theta_x_dot, document_motion=True)
        sim_theta2, sim_theta_dot2 = self.foil_chain.simulate(T, 10 ** 2, [0, 0, 0], initial_theta_x=initial_theta_x, initial_theta_x_dot=initial_theta_x_dot, document_motion=True)
        sim_theta3, sim_theta_dot3 = self.foil_chain.simulate(T, 10 ** 3, [0, 0, 0], initial_theta_x=initial_theta_x, initial_theta_x_dot=initial_theta_x_dot, document_motion=True)
        sim_theta4, sim_theta_dot4 = self.foil_chain.simulate(T, 10 ** 4, [0, 0, 0], initial_theta_x=initial_theta_x, initial_theta_x_dot=initial_theta_x_dot, document_motion=True)

        max_pos_diffs = np.array([np.max(np.abs(theoretical_theta[10 ** 3 - 1::10 ** 3] - sim_theta1.flatten())),
                                  np.max(np.abs(theoretical_theta[10 ** 2 - 1::10 ** 2] - sim_theta2.flatten())),
                                  np.max(np.abs(theoretical_theta[9::10] - sim_theta3.flatten())),
                                  np.max(np.abs(theoretical_theta - sim_theta4.flatten()))])

        max_vel_diffs = np.array([np.max(np.abs(theoretical_theta_dot[10 ** 3 - 1::10 ** 3] - sim_theta_dot1.flatten())),
                                  np.max(np.abs(theoretical_theta_dot[10 ** 2 - 1::10 ** 2] - sim_theta_dot2.flatten())),
                                  np.max(np.abs(theoretical_theta_dot[9::10] - sim_theta_dot3.flatten())),
                                  np.max(np.abs(theoretical_theta_dot - sim_theta_dot4.flatten()))])

        pos_convergence = (np.log10(max_pos_diffs)[0] - np.log10(max_pos_diffs)[np.log10(max_pos_diffs) > -14][-1]) / (np.sum(np.log10(max_pos_diffs) > -14) - 1)  # skip entries which are close to machine precision (expected to only be the last entry)
        vel_convergence = (np.log10(max_vel_diffs)[0] - np.log10(max_vel_diffs)[np.log10(max_vel_diffs) > -14][-1]) / (np.sum(np.log10(max_vel_diffs) > -14) - 1)  # skip entries which are close to machine precision (expected to only be the last entry)

        self.assertTrue(np.all(np.log10(max_pos_diffs) < -2))  # all errors should be small
        self.assertTrue(np.all(np.diff(np.log(max_pos_diffs)) < 0))  # guarantees that the error decreases
        self.assertTrue(np.all(np.diff(np.log(max_vel_diffs)) < 0))  # guarantees that the error decreases
        self.assertTrue(abs(pos_convergence - 4) < 0.05, msg=f'computed convergence rate is {pos_convergence}')  # guarantees that the convergence rate is as expected
        self.assertTrue(abs(vel_convergence - 4) < 0.05, msg=f'computed convergence rate is {vel_convergence}')  # guarantees that the convergence rate is as expected

    def test_simulation_double_pendulum_analytic(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'double_pendulum_analytic.txt'), base_path=base_path)

        # based on the analytic small angle solution for a much larger upper mass, taken from https://www.math.uwaterloo.ca/~sdalessi/EJP2023.pdf
        l_1 = 1
        l_2 = 0.5
        theta_1_0 = theta_2_0 = np.pi / 2 - 1.56  # analytical solution uses negative y axis as reference
        g = 9.80665
        t = (np.arange(200) + 1) / 200 * 20
        # non-dimensionalize the parameters
        L = l_2 / l_1
        tau = np.sqrt(g / l_1) * t

        theta_1_analytical = theta_1_0 * np.cos(tau)
        theta_2_analytical = (theta_2_0 - theta_1_0 / (1 - L)) * np.cos(tau / np.sqrt(L)) + theta_1_0 / (1 - L) * np.cos(tau)

        # Numerical FoilChain solution:
        theta_storage, _ = self.foil_chain.simulate(20, 200, np.array([[0, 0], [0, 0], [0, 0]]), document_motion=True)

        # we say that the test passes if the solutions are less than 5% appart, relative to the initial deflection
        theta_1_diff = np.abs(theta_1_analytical - (np.pi / 2 + theta_storage[0].flatten())) / theta_1_0
        theta_2_diff = np.abs(theta_2_analytical - (np.pi / 2 + theta_storage[1].flatten())) / theta_2_0

        self.assertTrue(np.max(theta_1_diff) < 0.05)
        self.assertTrue(np.max(theta_2_diff) < 0.05)

    def test_simulation_double_pendulum_external_sim(self):
        # TODO include in report with graphs. Comment about chaotic nature of system can lead to divergence if running for longer (from num error accumulation) but for the purpose at hand that is not an issue
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'double_pendulum_ext_sim.txt'), base_path=base_path)

        # based on the double pendulum simulation provided by MathWorks: https://uk.mathworks.com/help/symbolic/animation-and-solution-of-double-pendulum.html
        ref_data = np.loadtxt(join(base_path, 'Reference_Files', "double_pendulum_matlab.csv"), delimiter=',')
        ref_data = ref_data[[0, 2], 1:1001]

        # Numerical FoilChain solution:
        theta_storage, theta_dot_storage = self.foil_chain.simulate(0.1, 1000, np.array([[0, 0], [0, 0], [0, 0]]), document_motion=True)
        # We say that the test passes when the maximum absolute error is less than 1e-6

        theta_1_max_diff = max(np.abs(ref_data[1] - theta_storage[0] - np.pi / 2))
        theta_2_max_diff = max(np.abs(ref_data[0] - theta_storage[1] - np.pi / 2))

        self.assertLess(theta_1_max_diff, 1e-6)
        self.assertLess(theta_2_max_diff, 1e-6)

    def test_triple_pendulum_energy_conservation(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', 'triple_pendulum.txt'), base_path=base_path)
        theta_storage, theta_dot_storage = self.foil_chain.simulate(1, 100, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), document_motion=True)

        l_1 = self.foil_chain.foil_blocks[0].l_h
        m_1 = self.foil_chain.foil_blocks[0].m
        l_2 = self.foil_chain.foil_blocks[1].l_h
        m_2 = self.foil_chain.foil_blocks[1].m
        l_3 = self.foil_chain.foil_blocks[2].l_h
        m_3 = self.foil_chain.foil_blocks[2].m

        # The initial total energy is purely potential, with all masses at y=0
        total_E_0 = (m_1 + m_2 + m_3) * (l_1 + l_2 + l_3) * 9.81

        # Potential energy over time, computed relative to the height above the lowest attainable point of mass 3
        h1 = l_1 + l_2 + l_3 + l_1 * np.sin(theta_storage[0])
        h2 = h1 + l_2 * np.sin(theta_storage[1])
        h3 = h2 + l_3 * np.sin(theta_storage[2])

        PE_1 = m_1 * 9.81 * h1
        PE_2 = m_2 * 9.81 * h2
        PE_3 = m_3 * 9.81 * h3

        # Compute velocities and kinetic energy
        vx_1 = -l_1 * np.sin(theta_storage[0]) * theta_dot_storage[0]
        vy_1 = l_1 * np.cos(theta_storage[0]) * theta_dot_storage[0]

        vx_2 = vx_1 + -l_2 * np.sin(theta_storage[1]) * theta_dot_storage[1]
        vy_2 = vy_1 + l_2 * np.cos(theta_storage[1]) * theta_dot_storage[1]

        vx_3 = vx_2 + -l_3 * np.sin(theta_storage[2]) * theta_dot_storage[2]
        vy_3 = vy_2 + l_3 * np.cos(theta_storage[2]) * theta_dot_storage[2]

        KE_1 = 0.5 * m_1 * (vx_1 ** 2 + vy_1 ** 2)
        KE_2 = 0.5 * m_2 * (vx_2 ** 2 + vy_2 ** 2)
        KE_3 = 0.5 * m_3 * (vx_3 ** 2 + vy_3 ** 2)

        # Assess drift in total energy
        total_E = PE_1 + PE_2 + KE_1 + KE_2 + PE_3 + KE_3

        self.assertLess(np.log10(max(np.abs(total_E - total_E_0) / total_E_0)), -6)

    def test_geometry_transformation_computations(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', "geometry_transformation_test.txt"), base_path=base_path)
        self.foil_chain.theta_x = np.array([0.32, 0.15, -0.01])

        chord_position_vectors, left_hinge_position_vectors, rotated_profiles = self.foil_chain.export_transformed_geometry(testing=True)

        # Test by employing parts of the FoilChain class already verified in the testing of the init function
        np.testing.assert_array_almost_equal(left_hinge_position_vectors, [[0., 0.], [0.6712108,  0.22243215], [0.98346749, 0.17247454], [1.23141908, 0.2077592 ], [1.48200337, 0.21812816]], err_msg='Incorrect left_hinge_position_vectors')
        np.testing.assert_array_almost_equal(chord_position_vectors, [[-0.19683631, -0.03543257], [ 0.72054718,  0.23055134], [ 1.03645657,  0.16287469], [ 1.28105238,  0.21380366], [ 1.56655725,  0.29127624]], err_msg='Incorrect chord_position_vectors')
        self.assertEqual(len(rotated_profiles), 5, msg='incorrect rotated profile')
        np.testing.assert_array_equal(rotated_profiles[1][1000], [0, 0], err_msg='incorrect rotated profile')
        np.testing.assert_array_almost_equal(rotated_profiles[4], [[0.1 * np.cos(-0.3939498896722959), 0.1 * np.sin(-0.3939498896722959)], [0, 0], [0.1 * np.cos(-0.3939498896722959), 0.1 * np.sin(-0.3939498896722959)]], err_msg='incorrect rotated profile')

    def test_geometry_transformation_file_saving(self):
        self.foil_chain = FoilChain(join(base_path, 'Reference_Files', "geometry_transformation_test.txt"), base_path=base_path)
        self.foil_chain.theta_x = np.array([0.32, 0.15, -0.01])
        # Create testing directory
        os.mkdir(join(base_path, 'unit_test_temporary'))

        _, _ = self.foil_chain.export_transformed_geometry(chord_position_file_name='unit_test_temporary/chord_position.txt', profile_base_name='unit_test_temporary/airfoil')

        # Check if all files have been created correctly
        self.assertTrue(os.path.isfile(join(base_path, 'unit_test_temporary/chord_position.txt')), msg='Chord position file not found')
        for i in range(5):
            self.assertTrue(os.path.isfile(join(base_path, f'unit_test_temporary/airfoil_{i}.dat')), msg='profile file not found')

        np.testing.assert_array_almost_equal(np.loadtxt(join(base_path, 'unit_test_temporary/airfoil_4.dat')), [[0.1 * np.cos(-0.3939498896722959), 0.1 * np.sin(-0.3939498896722959)], [0, 0], [0.1 * np.cos(-0.3939498896722959), 0.1 * np.sin(-0.3939498896722959)]], err_msg='profile file incorrect')

        # Clean up by deleting all files
        os.remove(join(base_path, 'unit_test_temporary/chord_position.txt'))
        for i in range(5):
            os.remove(join(base_path, f'unit_test_temporary/airfoil_{i}.dat'))

        os.removedirs(join(base_path, 'unit_test_temporary'))


class TestFixedFoilChain(unittest.TestCase):
    """
    Testing the FixedFoilChain class in fixed_foil_chain.py
    """

    def setUp(self):
        self.fixed_foil_chain = FixedFoilChain(join(base_path, 'Reference_Files', 'init_test_fixed_airfoils.txt'), 2, base_path=base_path)

    def test_init(self):
        self.assertEqual(self.fixed_foil_chain.n_airfoils, 2)
        np.testing.assert_array_almost_equal(self.fixed_foil_chain.last_rh_pos, [0.56294184, 0.06032596])
        np.testing.assert_array_equal(self.fixed_foil_chain.theta_x_dot, [0, 0])

        # check that theta_x and theta_x_dot are indeed read-only
        with self.assertRaises(ValueError):
            self.fixed_foil_chain.theta_x[1] = 0

        with self.assertRaises(ValueError):
            self.fixed_foil_chain.theta_x_dot[1] = 0
