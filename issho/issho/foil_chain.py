import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from .airfoil import Airfoil
from .foil_block import FoilBlock
from .helper import rotate

# TODO remove this
import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('macosx')


class FoilChain:
    """
    Class handling the dynamics of a string of connected airfoils
    """

    def __init__(self, input_file, base_path=''):
        """
        :param input_file: name of input text file, which must follow the specified structure
        """
        self.base_path = base_path  # Absolute path to the place from which files are to be loaded and to which files are to be stored
        with open(os.path.join(base_path, input_file)) as file:
            input_lines = file.readlines()

        n_fixed_airfoils = int(input_lines[6].split(' ')[0])
        self.n_airfoils = int(input_lines[8].split(' ')[0]) - n_fixed_airfoils
        self.g = float(input_lines[9].split(', ')[0])  # strength of gravitational field
        self.theta_g = float(input_lines[9].split(' ')[1])  # cc-positive angle of gravitational field relative to negative y-axis
        if len(input_lines) > 19 + 2 * (self.n_airfoils + n_fixed_airfoils):
            warnings.warn(f"Mismatch detected between specified number of airfoils and length of input file. Only the first {self.n_airfoils + n_fixed_airfoils} airfoils and hinges will be considered")

        if len(input_lines) < 19 + 2 * (self.n_airfoils + n_fixed_airfoils):
            raise Exception("Specified number of airfoils is greater than the number of provided airfoil and hinge lines")

        if self.n_airfoils == 0:
            raise Exception("No moveable airfoils! FoilChain should not be invoked if only one fixed FoilBlock is being simulated")

        # initialize storage
        self.airfoils = []  # Stores all created Airfoil objects
        self.foil_blocks = []  # Stores all created FoilBlock objects
        self.foil_block_indices_per_block = []  # Stores the indices of the airfoils composing the corresponding FoilBlock
        self.foil_block_indices_per_airfoil = []  # Stores the indices of the FoilBlock that the individual airfoils belong to
        foil_block_relative_angles = []  # Store the angles at which the FoilBlock hinge lines are initially placed relative to one another. First value is relative to the positive x-axis
        foil_block_hinge_lengths = []  # hinge lengths of the foil blocks
        foil_block_cm_lengths = []  # cm lengths of the foil blocks
        foil_block_cm_thetas = []  # theta_cm for the foil blocks
        foil_block_masses = []  # masses of the FoilBlocks
        foil_block_Js = []  # polar mass moments of inertia of the FoilBlocks
        spring_matrix = np.zeros((self.n_airfoils, self.n_airfoils + 1))  # one additional column to simplify adding the first hinge
        spring_moment_vector = np.zeros(self.n_airfoils)
        damping_matrix = np.zeros((self.n_airfoils, self.n_airfoils + 1))  # one additional column to simplify adding the first hinge
        self.all_l_h = []  # hinge lengths for all component airfoils
        self.airfoil_left_hinge_vectors = []  # vector of left hinge in the airfoil coordinate system
        self.profiles_x = []  # Long concatenated list of all airfoil profile x coordinates
        self.profiles_y = []  # Long concatenated list of all airfoil profile y coordinates
        self.profile_indices = [0]  # Cumulative sum of profile lengths, i.e., giving the indices of the different profiles in the long concatenated array of profiles
        self.additional_rotation_vector_hinge = []  # Angle of airfoil hinge line (i.e. profile x-axis) relative to FoilBlock hinge line (= 0 OR (if n_airfoil>1) = - FoilBlock.theta_h + FoilBlock.components[0].theta_h + cumsum([0, FoilBlock.component_angles])[i])
        self.additional_rotation_vector_chord = []  # Angle of airfoil chord line (i.e. profile x-axis) relative to FoilBlock hinge line (= -airfoil.theta_h OR (if n_airfoil>1) = -FoilBlock.components[i].theta_h - FoilBlock.theta_h + FoilBlock.components[0].theta_h + cumsum([0, FoilBlock.component_angles])[i])

        initial_fixed_angle = 0  # find total angle for fixed airfoils
        for i in range(n_fixed_airfoils):
            initial_fixed_angle += float(input_lines[19 + 2 * i].strip().split(', ')[2])

        fixed_hinge = False  # Boolean to discern when airfoils are fixed together in one FoilBlock
        next_angle_correction = 0  # Correction increment to account for the possible difference between airfoil and FoilBlock hinge line angles
        total_chord_length = 0
        for i in range(self.n_airfoils):
            # create the relevant airfoil and store it
            current_hinge_inpt = input_lines[19 + 2 * (i + n_fixed_airfoils)].strip().split(', ')  # two times i such that we skip the airfoils

            next_hinge_inpt = [None]
            if i < self.n_airfoils - 1:
                next_hinge_inpt = input_lines[21 + 2 * (i + n_fixed_airfoils)].strip().split(', ')  # two times i such that we skip the airfoils

            if i == 0 and current_hinge_inpt[0] == 'fixed':
                raise Exception("The first hinge may not be fixed. Simply exclude the airfoil from the FoilChain")

            local_foil_inpt = input_lines[20 + 2 * (i + n_fixed_airfoils)].strip().split(', ')  # two times i such that we skip the hinges
            local_airfoil = Airfoil(local_foil_inpt[0], [float(local_foil_inpt[2]), float(local_foil_inpt[3])], [float(local_foil_inpt[4]), float(local_foil_inpt[5])], [float(local_foil_inpt[6]), float(local_foil_inpt[7])], float(local_foil_inpt[8]), float(local_foil_inpt[9]), l_chord=float(local_foil_inpt[1]), profile=None, base_path=base_path)
            total_chord_length += local_airfoil.l_chord
            self.airfoils.append(local_airfoil)
            self.all_l_h.append(local_airfoil.l_h)
            self.airfoil_left_hinge_vectors.append(local_airfoil.pos_lh)
            self.profiles_x = np.concatenate((self.profiles_x, local_airfoil.profile[:, 0].flatten()), dtype=float)
            self.profiles_y = np.concatenate((self.profiles_y, local_airfoil.profile[:, 1].flatten()), dtype=float)
            self.foil_block_indices_per_airfoil.append(len(self.foil_blocks))
            self.profile_indices.append(len(local_airfoil.profile))

            if not fixed_hinge:
                # i.e., current airfoil is NOT fixed to previous one, so the lists for a new FoilBlock must be setup
                local_airfoil_list = []
                local_airfoil_index_list = []
                local_hinge_angle_list = []
                local_airfoil_theta_h_list = []

                # Store information about the current hinge, which will preceed the to-be-created FoilBlock
                foil_block_relative_angles.append(float(current_hinge_inpt[2]) - next_angle_correction)

                if current_hinge_inpt[0] == 'spring':
                    spring_constant = float(current_hinge_inpt[1])
                    if spring_constant < 0:
                        raise Exception('A negative spring constant will lead to unstable and non-physical behaviour')
                    block_idx = len(self.foil_blocks)
                    spring_matrix[block_idx, block_idx : block_idx + 2] = [-spring_constant, spring_constant]
                    spring_moment_vector[block_idx] = - spring_constant * float(current_hinge_inpt[3])

                elif current_hinge_inpt[0] == 'damped':
                    damping_coeff = float(current_hinge_inpt[1])
                    if damping_coeff < 0:
                        raise Exception('A negative damping coefficient will lead to unstable and non-physical behaviour')
                    block_idx = len(self.foil_blocks)
                    damping_matrix[block_idx, block_idx: block_idx + 2] = [-damping_coeff, damping_coeff]

                elif current_hinge_inpt[0] == 'spring_damped':
                    spring_constant = float(current_hinge_inpt[1])
                    damping_coeff = float(current_hinge_inpt[4])
                    if spring_constant < 0 or damping_coeff < 0:
                        raise Exception('A negative spring constant or damping coefficient will lead to unstable and non-physical behaviour')

                    block_idx = len(self.foil_blocks)
                    spring_matrix[block_idx, block_idx: block_idx + 2] = [-spring_constant, spring_constant]
                    spring_moment_vector[block_idx] = - spring_constant * float(current_hinge_inpt[3])
                    damping_matrix[block_idx, block_idx: block_idx + 2] = [-damping_coeff, damping_coeff]

                elif current_hinge_inpt[0] != 'pin':
                    # we know that the hinge also isn't fixed, so an invalid hinge type must have been provided
                    raise Exception('Invalid type of hinge provided!\n\n Valid hinge types are: pin, spring, damped, or fixed. Please make sure you spelled everything correctly and lowercase')

            fixed_hinge = next_hinge_inpt[0] == 'fixed'  # True if the next airfoil is fixed to the current one

            local_airfoil_list.append(local_airfoil)  # store the current airfoil as part of the current FoilBlock
            local_airfoil_index_list.append(i)  # store the index of the current airfoil as belonging to the current FoilBlock
            local_airfoil_theta_h_list.append(local_airfoil.theta_h)  # Store the angle of the current airfoil hinge line relative to its own chord line

            if fixed_hinge:
                local_hinge_angle_list.append(float(next_hinge_inpt[2]))  # Store the internal angle of components within the FoilBlock
            else:
                # Next hinge is not fixed, so we can create a FoilBlock from the current storage lists
                local_foil_block = FoilBlock(local_airfoil_list, local_hinge_angle_list)

                # Store all relevant properties of the FoilBlock
                self.foil_block_indices_per_block.append(local_airfoil_index_list)
                self.foil_blocks.append(local_foil_block)
                foil_block_hinge_lengths.append(local_foil_block.l_h)
                foil_block_cm_lengths.append(local_foil_block.l_cm)
                foil_block_cm_thetas.append(local_foil_block.theta_cm)
                foil_block_masses.append(local_foil_block.m)
                foil_block_Js.append(local_foil_block.J)

                # If the FoilBlock consits of multiple components, there may be a difference between the hinge lines of the overall FoilBlock and the individual aifoils
                # This is an issue as the input file specifies the hinge angles relative to the hinge lines of the individual airfoils. Thus, we need to incorporate the correction factors
                foil_block_relative_angles[-1] += local_foil_block.first_hinge_angle_diff  # first_hinge_angle_diff gives the cc-positive angle from hinge line of first airfoil to overall hinge line
                next_angle_correction = local_foil_block.last_hinge_angle_diff  # last_hinge_angle_diff gives the cc-positive angle from hinge line of last airfoil to overall hinge line

                # Finally store the additional rotation vector for that FoilBlock
                self.additional_rotation_vector_hinge = np.concatenate((self.additional_rotation_vector_hinge, - local_foil_block.theta_h + local_foil_block.components[0].theta_h + np.cumsum(np.concatenate(([0], local_foil_block.component_angles)))))
                self.additional_rotation_vector_chord = np.concatenate((self.additional_rotation_vector_chord, -np.array(local_airfoil_theta_h_list) - local_foil_block.theta_h + local_foil_block.components[0].theta_h + np.cumsum(np.concatenate(([0], local_foil_block.component_angles)))))

        if not np.isclose(total_chord_length, 1, 4):
            # Typically a total chord length of 1 is required
            warnings.warn(f"The total chord length of all considered airfoils is {total_chord_length}. Usually a total chord length of 1 is recommended in UTCart")

        # create the arrays necessary for the dynamics computations
        self.n_foilblocks = len(self.foil_blocks)
        self.theta_x = np.cumsum(foil_block_relative_angles) + initial_fixed_angle  # Initial cc-positive angles of FoilBlock hinge lines relative to the positive x-axis
        self.theta_x_dot = np.zeros(self.n_foilblocks)  # The FoilChain is initially set as being at rest
        self.all_l_h = np.array(self.all_l_h)
        self.airfoil_left_hinge_vectors = np.array(self.airfoil_left_hinge_vectors)
        self.l_h = np.array(foil_block_hinge_lengths)
        self.l_cm = np.array(foil_block_cm_lengths)
        self.theta_cm = np.array(foil_block_cm_thetas)
        self.m = np.array(foil_block_masses)
        self.J = np.array(foil_block_Js)
        self.foil_block_indices_per_airfoil = np.array(self.foil_block_indices_per_airfoil)
        self.foil_block_first_airfoil_indices = np.maximum.accumulate(np.concatenate(([1], self.foil_block_indices_per_airfoil[1:] - self.foil_block_indices_per_airfoil[:-1])) * np.arange(self.n_airfoils))  # For each airfoil, the index of the first airfoil in the same FoilBlock
        self.profile_indices = np.cumsum(self.profile_indices)
        self.profile_coords = np.vstack((self.profiles_x, self.profiles_y)).T

        # create the matrix for computing the hinge moments, s.t. moments = matrix @ [theta_x, theta_x_dot] + spring_moment_vector
        self.moment_computation_matrix = np.hstack((spring_matrix[:self.n_foilblocks, 1 : self.n_foilblocks + 1], damping_matrix[:self.n_foilblocks, 1 : self.n_foilblocks + 1]))  # Concatenate the truncated spring and damping matrices
        self.spring_moment_vector = np.array(spring_moment_vector[:self.n_foilblocks])

    def __create_simulation_matrix(self, theta_x=None):
        # Private method that creates the matrix necessary for computing the reaction forces and angular accelerations
        # based on a vector of unknowns of the format [... R_x ... R_y ... theta_x_ddot]
        
        if theta_x is None:
            theta_x = self.theta_x

        raw_matrix = np.zeros((3 * self.n_foilblocks, 3 * (self.n_foilblocks + 1)))  # add three additional columns, which correspond to the -1th FoilBlock and will ultimately be truncated

        # To ease matrix construction we create auxiliary local arrays. The added values are irrelevant but are set to the physically realistic values
        lcl_m = np.concatenate(([np.inf], self.m))  # first aifoil is attached to something unmovable, i.e., something of infinite mass
        lcl_l_cm = np.concatenate(([0], self.l_cm))  # zero as there is no left hinge
        lcl_l_h = np.concatenate(([0], self.l_h))  # zero as there is no left hinge
        lcl_theta_cm = np.concatenate(([0], self.theta_cm))  # zero as there is no cm defined
        lcl_theta_x = np.concatenate(([0], theta_x))  # zero as there is no left hinge
        lcl_total_angle = lcl_theta_x + lcl_theta_cm

        # Equation 1, from acceleration of the cm in the x-direction
        raw_matrix[: self.n_foilblocks, : self.n_foilblocks] += np.diag(1 / lcl_m[:-1])  # For the R_x_{i-1} term: 1 / m_{i-1}
        raw_matrix[: self.n_foilblocks, 1 : self.n_foilblocks + 1] += -(np.diag(1 / lcl_m[:-1] + 1 / lcl_m[1:]))  # For the R_x_i term: -(1 / m_{i-1} + 1 / m_{i})
        raw_matrix[: self.n_foilblocks -1, 2 : self.n_foilblocks + 1] += np.diag(1 / lcl_m[1 : -1])  # For the R_x_{i+1} term. The last airfoil has no i + 1. The term is 1 / m_{i}

        raw_matrix[: self.n_foilblocks, 2 * (self.n_foilblocks + 1) : -1] += np.diag(- (lcl_l_cm[:-1] * np.sin(lcl_total_angle[:-1]) - lcl_l_h[:-1] * np.sin(lcl_theta_x[:-1])))  # For the theta_ddot_x_{i-1} term: -(l_cm_{i-1} sin(theta_cm_{i-1} + theta_x_{i-1}) -l_h_{i-1} sin(theta_x_{i-1})
        raw_matrix[: self.n_foilblocks, 2 * (self.n_foilblocks + 1) + 1: ] += np.diag(lcl_l_cm[1:] * np.sin(lcl_total_angle[1:]))  # For the theta_ddot_x_{i} term: l_cm_{i} sin(theta_cm_{i} + theta_x_{i})

        # Equation 2, from acceleration of the cm in the y-direction
        raw_matrix[self.n_foilblocks : 2 * self.n_foilblocks, self.n_foilblocks + 1 : 2 * self.n_foilblocks + 1] += np.diag(1 / lcl_m[:-1])  # For the R_y_{i-1} term: 1 / m_{i-1}
        raw_matrix[self.n_foilblocks : 2 * self.n_foilblocks, self.n_foilblocks + 2 : 2 * self.n_foilblocks + 2] += -(np.diag(1 / lcl_m[:-1] + 1 / lcl_m[1:]))  # For the R_y_i term: -(1 / m_{i-1} + 1 / m_{i})
        raw_matrix[self.n_foilblocks : 2 * self.n_foilblocks -1, self.n_foilblocks + 3 : 2 * self.n_foilblocks + 2] += np.diag(1 / lcl_m[1: -1])  # For the R_y_{i+1} term. The last airfoil has no i + 1. The term is 1 / m_{i}

        raw_matrix[self.n_foilblocks : 2 * self.n_foilblocks, 2 * (self.n_foilblocks + 1): -1] += np.diag(- (lcl_l_h[:-1] * np.cos(lcl_theta_x[:-1]) - lcl_l_cm[:-1] * np.cos(lcl_total_angle[:-1])))  # For the theta_ddot_x_{i-1} term: -(l_h_{i-1} cos(theta_x_{i-1} - l_cm_{i-1} cos(theta_cm_{i-1} + theta_x_{i-1}))
        raw_matrix[self.n_foilblocks : 2 * self.n_foilblocks, 2 * (self.n_foilblocks + 1) + 1:] += np.diag(- lcl_l_cm[1:] * np.cos(lcl_total_angle[1:]))  # For the theta_ddot_x_{i} term: - l_cm_{i} cos(theta_cm_{i} + theta_x_{i})

        # Equation 3, from angular acceleration
        raw_matrix[2 * self.n_foilblocks :, 1 : self.n_foilblocks + 1] += np.diag(-lcl_l_cm[1:] * np.sin(lcl_total_angle[1:]))  # For the R_x_i term: -l_cm_{i} sin(theta_x_{i} + theta_cm_{i})
        raw_matrix[2 * self.n_foilblocks :-1, 2: self.n_foilblocks + 1] += np.diag(lcl_l_cm[1:-1] * np.sin(lcl_total_angle[1:-1]) - lcl_l_h[1:-1] * np.sin(lcl_theta_x[1:-1]))  # For the R_x_{i+1} term: l_cm_{i} sin(theta_x_{i} + theta_cm_{i}) - l_h_{i} sin(theta_x_{i})

        raw_matrix[2 * self.n_foilblocks :, self.n_foilblocks + 2 : 2 * self.n_foilblocks + 2] += np.diag(lcl_l_cm[1:] * np.cos(lcl_total_angle[1:]))  # For the R_x_i term: l_cm_{i} cos(theta_x_{i} + theta_cm_{i})
        raw_matrix[2 * self.n_foilblocks :-1, self.n_foilblocks + 3 : 2 * self.n_foilblocks + 2] += np.diag(lcl_l_h[1:-1] * np.cos(lcl_theta_x[1:-1]) - lcl_l_cm[1:-1] * np.cos(lcl_total_angle[1:-1]))  # For the R_x_{i+1} term: l_h_{i} cos(theta_x_{i}) - l_cm_{i} cos(theta_x_{i} + theta_cm_{i})

        raw_matrix[2 * self.n_foilblocks :, 2 * (self.n_foilblocks + 1) + 1:] += -np.diag(self.J)  # For the theta_ddot_x_{i} term: J_i

        # return the matrix without the auxilliary columns
        return raw_matrix[:, np.arange(3 * (self.n_foilblocks + 1)) % (self.n_foilblocks + 1) != 0]

    def __create_forcing_vector(self, F_x, F_y, M, theta_x=None, theta_x_dot=None):
        """
        Private method that creates the forcing vector necessary for computing the reaction forces and angular accelerations
        Uses the angular positions and velocities currently attached to self.theta_x and self.theta_x_dot respectively

        :param F_x: array of length self.n_foilblock containing the forces in the x-direction acting on the left hinge of each FoilBlock
        :param F_y: array of length self.n_foilblock containing the forces in the y-direction acting on the left hinge of each FoilBlock
        :param M: array of length self.n_foilblock containing the moments acting around the left hinge of each FoilBlock
        :param theta_x: if None, the theta_x currently attached to the FoilChain will be used
        :param theta_x_dot: if None, the theta_x_dot currently attached to the FoilChain will be used
        :return: array of length 3 * self.n_foilblock
        """
        
        if theta_x is None:
            theta_x = self.theta_x
        if theta_x_dot is None:
            theta_x_dot = self.theta_x_dot

        forcing_vector = np.zeros(self.n_foilblocks * 3)  # initialize vector
        total_angle = theta_x + self.theta_cm  # to make the code more readable

        # Adjust inputs to account for gravitational effect
        F_y = F_y - self.m * self.g * np.cos(self.theta_g)
        F_x = F_x + self.m * self.g * np.sin(self.theta_g)
        M = M + self.l_cm * self.m * self.g * (np.cos(total_angle) * np.cos(self.theta_g) + np.sin(total_angle) * np.sin(self.theta_g))

        # Compute hinge moments using moments = matrix @ [theta_x, theta_x_dot] + spring_moment_vector
        M_hinge = self.moment_computation_matrix @ np.concatenate((theta_x, theta_x_dot)) + self.spring_moment_vector  # ravel to flatten and avoid shape mismatches

        # Equation 1, from acceleration of the cm in the x-direction
        forcing_vector[: self.n_foilblocks] = - 1 / self.m * F_x - theta_x_dot ** 2 * self.l_cm * np.cos(total_angle)  # Terms related to index i
        forcing_vector[1 : self.n_foilblocks] += 1 / self.m[:-1] * F_x[:-1] + theta_x_dot[:-1] ** 2 * (self.l_cm[:-1] * np.cos(total_angle[:-1]) - self.l_h[:-1] * np.cos(theta_x[:-1]))  # Terms related to index i-1 (does not apply to the first FoilBlock)

        # Equation 2, from acceleration of the cm in the y-direction
        forcing_vector[self.n_foilblocks : 2 * self.n_foilblocks] = - 1 / self.m * F_y - theta_x_dot ** 2 * self.l_cm * np.sin(total_angle)  # Terms related to index i
        forcing_vector[self.n_foilblocks + 1: 2 * self.n_foilblocks] += 1 / self.m[:-1] * F_y[:-1] + theta_x_dot[:-1] ** 2 * (self.l_cm[:-1] * np.sin(total_angle[:-1]) - self.l_h[:-1] * np.sin(theta_x[:-1]))  # Terms related to index i-1 (does not apply to the first FoilBlock)

        # Equation 3, from angular acceleration
        forcing_vector[self.n_foilblocks * 2 :] = M + M_hinge - self.l_cm * np.sin(total_angle) * F_x + self.l_cm * np.cos(total_angle) * F_y  # Terms related to index i
        forcing_vector[self.n_foilblocks * 2: - 1] += - M_hinge[1:]  # Terms related to index i+1 (does not apply to the last FoilBlock, which has no right hinge)

        return forcing_vector

    def simulate(self, T, n_t, forces, initial_theta_x=None, initial_theta_x_dot=None, force_file=None, document_motion=False, document_hinge_forces=False, progress_bar=True):
        # Handle the inputs
        dt = T / n_t

        if initial_theta_x is not None:
            self.theta_x = np.ravel([initial_theta_x])

        if initial_theta_x_dot is not None:
            self.theta_x_dot = np.ravel([initial_theta_x_dot])

        if document_motion:
            # To avoid overlap when doing multiple simulations, it does not include the starting conditions (we should know what they are)
            theta_storage = np.zeros((self.n_foilblocks, n_t))
            theta_dot_storage = np.zeros((self.n_foilblocks, n_t))

        if document_hinge_forces:
            hinge_force_storage = np.zeros((2 * self.n_foilblocks, n_t))

        if forces is None:
            # In this case a force_file must have been specified
            if force_file is None:
                raise Exception("If forces is None, a valid file must be provided for the keyward argument 'force_file'")
            forces = np.loadtxt(force_file)

        F_x = np.ravel([forces[0]])
        F_y = np.ravel([forces[1]])
        M = np.ravel([forces[2]])

        # Integration loop based on the RK4 method
        for i in tqdm(range(n_t), disable=not progress_bar):
            # Calculate the angular acceleration based on the current velocity and position
            sim_matrix = self.__create_simulation_matrix()
            forcing_vector = self.__create_forcing_vector(F_x, F_y, M)
            solution_vector = np.linalg.solve(sim_matrix, forcing_vector)
            theta_x_ddot_0 = solution_vector[-self.n_foilblocks:]

            if document_hinge_forces:
                hinge_force_storage[:, i] = solution_vector[:- self.n_foilblocks]

            # compute intermediate stages
            k0 = dt * theta_x_ddot_0
            theta_x_1 = self.theta_x + 0.5 * self.theta_x_dot * dt
            theta_x_dot_1 = self.theta_x_dot + k0 / 2
            k1 = dt * np.linalg.solve(self.__create_simulation_matrix(theta_x=theta_x_1), self.__create_forcing_vector(F_x, F_y, M, theta_x=theta_x_1, theta_x_dot=theta_x_dot_1))[-self.n_foilblocks:]
            theta_x_2 = theta_x_1 + dt / 4 * k0
            theta_x_dot_2 = self.theta_x_dot + k1 / 2
            k2 = dt * np.linalg.solve(self.__create_simulation_matrix(theta_x=theta_x_2), self.__create_forcing_vector(F_x, F_y, M, theta_x=theta_x_2, theta_x_dot=theta_x_dot_2))[-self.n_foilblocks:]
            theta_x_3 = self.theta_x + self.theta_x_dot * dt + dt / 2 * k1
            theta_x_dot_3 = self.theta_x_dot + k2
            k3 = dt * np.linalg.solve(self.__create_simulation_matrix(theta_x=theta_x_3), self.__create_forcing_vector(F_x, F_y, M, theta_x=theta_x_3, theta_x_dot=theta_x_dot_3))[-self.n_foilblocks:]

            # Update position and velocity
            self.theta_x = self.theta_x + dt * self.theta_x_dot + dt / 6 * (k0 + k1 + k2)
            self.theta_x_dot = self.theta_x_dot + (k0 + 2 * k1 + 2 * k2 + k3) / 6

            # If applicable document the new position and velocity
            if document_motion:
                theta_storage[:, i] = self.theta_x
                theta_dot_storage[:, i] = self.theta_x_dot

        # If storing the hinge forces we need to do one more force computation
        if document_hinge_forces:
            sim_matrix = self.__create_simulation_matrix()
            forcing_vector = self.__create_forcing_vector(F_x, F_y, M)
            hinge_force_storage[:, -1] = np.linalg.solve(sim_matrix, forcing_vector)[:- self.n_foilblocks]

        if document_motion or document_hinge_forces:
            return_list = []
            if document_motion:
                return_list.append(theta_storage)
                return_list.append(theta_dot_storage)
            if document_hinge_forces:
                return_list.append(hinge_force_storage)

            return return_list

    def animate_simulation(self, theta_storage):
        fig, ax = plt.subplots()

        # Initialize the line
        lines = []
        for i in range(self.n_foilblocks):
            lines.append(ax.plot([], [], marker='o')[0])

        # Set axis limits
        ax.set_xlim(-self.l_h.sum(), self.l_h.sum())
        ax.set_ylim(-self.l_h.sum(), self.l_h.sum())

        # Define the animation function
        def animate(frame):
            for i in range(self.n_foilblocks):
                if i == 0:
                    x_l = 0
                    y_l = 0
                else:
                    x_l = x_r
                    y_l = y_r

                x_r = x_l + self.l_h[i] * np.cos(theta_storage[i, frame])
                y_r = y_l + self.l_h[i] * np.sin(theta_storage[i, frame])

                lines[i].set_data([x_l, x_r], [y_l, y_r])
            return lines,

        # Create the animation
        ani = FuncAnimation(fig, animate, frames=np.arange(theta_storage.shape[1]), interval=1)

        # Show the animation
        plt.show()

    def export_transformed_geometry(self, chord_position_file_name=None, profile_base_name='Simulation/airfoil', profile_scale_factor=1, testing=False):
        """
        The function computes and saves the rotated airfoil profiles, as well as the positions of the airfoil chords.
        IMPORTANT: The reference point for these position vectors is the left hinge of the left-most airfoil, which is set to be at a location (0, 0)
        """

        # Compute the angular position of the airfoil hinge and chord lines in the overall reference frame
        chord_line_angles = self.theta_x[self.foil_block_indices_per_airfoil] + self.additional_rotation_vector_chord  # angle from x-axis to hinge line plus angle from hinge line to chord line
        hinge_line_angles = self.theta_x[self.foil_block_indices_per_airfoil] + self.additional_rotation_vector_hinge  # angle from x-axis to hinge line

        # Compute the locations of the airfoil left hinges and chord line beginnings in the overall reference frame
        left_hinge_position_vectors = self.all_l_h[:-1].reshape(-1, 1) * np.hstack((np.cos(hinge_line_angles[:-1]).reshape(-1, 1), np.sin(hinge_line_angles[:-1]).reshape(-1, 1)))
        left_hinge_position_vectors = np.vstack((np.zeros((1, 2)), np.cumsum(left_hinge_position_vectors, axis=0)))

        chord_position_vectors = left_hinge_position_vectors - rotate(self.airfoil_left_hinge_vectors.T, chord_line_angles).T  # left hinge position minus vector from the chord base point to the left hinge
        if chord_position_file_name is not None:
            np.savetxt(os.path.join(self.base_path, chord_position_file_name), chord_position_vectors)

        # Compute and save the new airfoil profiles
        rotated_profiles = []
        for i in range(self.n_airfoils):
            rotated_profiles.append(rotate(self.profile_coords[self.profile_indices[i] : self.profile_indices[i + 1]].T, chord_line_angles[i]).T * profile_scale_factor)
            if not testing:
                np.savetxt(os.path.join(self.base_path, profile_base_name + f'_{i}.dat'), rotated_profiles[i])

        if testing:
            return chord_position_vectors, left_hinge_position_vectors, rotated_profiles

        else:
            return chord_position_vectors, left_hinge_position_vectors
