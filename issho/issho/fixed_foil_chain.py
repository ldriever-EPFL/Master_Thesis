import os
import warnings
import numpy as np
from .foil_chain import FoilChain


class FixedFoilChain(FoilChain):
    """
    Class for potential strings of fixed airfoils. Behaves like a FoilChain but cannot be moved
    """

    def __init__(self, input_file, n_fixed_airfoils, base_path=''):
        # read the original input file
        with open(os.path.join(base_path, input_file)) as file:
            input_lines = file.readlines()

        if n_fixed_airfoils != int(input_lines[6].split(' ')[0]):
            warnings.warn(f"\nThe used number of fixed airfoils is {n_fixed_airfoils}, which is different from {int(input_lines[6].split(' ')[0])}, the number originally specified in the input file!\n")

        input_lines[6] = '0\n'  # pretend that there are no fixed airfoils
        input_lines[8] = f'{n_fixed_airfoils}\n'  # pretend that there are no fixed airfoils
        input_lines[19] = f"pin, None, {float(input_lines[19].strip().split(', ')[2])}\n"  # modify the first hinge to be of pin-type
        input_lines = input_lines[:19 + 2 * n_fixed_airfoils]  # Truncate all the non-fixed airfoils

        # create the temporary input file
        with open(os.path.join(base_path, 'temp_fc_input.txt'), 'w') as file:
            file.writelines(input_lines)

        # Use the FoilChain init method
        try:
            super(FixedFoilChain, self).__init__('temp_fc_input.txt', base_path=base_path)
        except:
            # clean up in case that the init fails
            os.remove(os.path.join(base_path, 'temp_fc_input.txt'))
            raise

        # Remove the temporary file
        os.remove(os.path.join(base_path, 'temp_fc_input.txt'))

        # make theta_x read-only
        self.theta_x.flags.writeable = False
        self.theta_x_dot.flags.writeable = False

        # store the right-hinge_coordinates of the last airfoil, relative to the left-most hinge (which is placed at (0, 0))
        hinge_line_angles = self.theta_x[self.foil_block_indices_per_airfoil] + self.additional_rotation_vector_hinge  # angle from x-axis to hinge line
        right_hinge_position_vectors = self.all_l_h.reshape(-1, 1) * np.hstack((np.cos(hinge_line_angles).reshape(-1, 1), np.sin(hinge_line_angles).reshape(-1, 1)))
        self.last_rh_pos = np.sum(right_hinge_position_vectors, axis=0)

    # Overwrite the simulation and animation methods to avoid misuse of the FixedFoilChain class:
    def simulate(self, *args, **kwargs):
        warnings.warn("This is the FixedFoilChain class - simulation of motion is not possible")

    def animate_simulation(self, *args, **kwargs):
        warnings.warn("This is the FixedFoilChain class - animation of motion is not possible")
