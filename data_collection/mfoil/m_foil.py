import numpy as np
from .helper import airfoil_splitter, airfoil_plotter, rotate, get_2pi_vec_angle, sever_rear


class MFoil:
    """
    MFoil - the M is for multiple - is a class for splitting an airfoil into sections, rotating
    the sections as desired, and saving the rotated profiles to files...
    """

    def __init__(self, base_foil_file, chord_fractions, hinge_positions, nose_up_thetas, nose_down_thetas, straight_rear=False, smoothen_tip=False, N_arc_points=1000, theta_lim_deg=5, base_name='foil', do_ref_split=False):
        """

        :param base_foil_file: file of un-split airfoil
        :param chord_fractions: chord fractions giving the lengths of the first n-1 airfoils (the last one gets what is left
        :param hinge_positions: hinge positions relative to the corresponding splitting point, as defined as a result of #chord_fractions'
        :param nose_up_thetas: maximum clockwise deflection for a foil section
        :param nose_down_thetas: maximum counter-clockwise deflection for a foil section
        :param straight_rear: Bool. If True, the rear of the foil sections is vertically straight
        :param N_arc_points: NUmber of points used to discretize the cutting arc
        :param theta_lim_deg: Minimum angle between two adjacent tangents for a point to count as a kink
        :param do_ref_split: Bool. If True, the area close to the split-induced kinks is saved as separate profile, allowing for additional refinement
        """

        self.base_foil = np.loadtxt(base_foil_file)
        self.base_name = base_name
        if abs(max(self.base_foil[:, 0]) - min(self.base_foil[:, 0]) - 1) > 1e-2:
            raise Exception('base foil must have a chord length of 1')
        self.chord_fractions = np.array(chord_fractions)
        if np.sum(chord_fractions) >= 1:
            raise Exception('chord fraction should add up to less than 1')
        self.hinge_positions = np.array(hinge_positions)
        self.nose_up_thetas = np.array(nose_up_thetas)
        self.nose_down_thetas = np.array(nose_down_thetas)
        self.n_foils = np.size(chord_fractions) + 1
        self.do_ref_split = do_ref_split
        self.rear_ref_split_indices = []  # Indices indicating the rear part of an airfoil section, which is to be split off for refinement
        self.front_ref_aux_angles = []  # Indices indicating the top and bottom-most points of the arc on the front of an airfoil section

        if np.any(np.array([np.size(hinge_positions) / 2, np.size(nose_up_thetas), np.size(nose_down_thetas)]) != self.n_foils - 1):
            raise Exception('Inconsistent input array sizes. Number of implied airfoils is not consistend')

        # Create the split foil sections
        self.split_lengths = np.cumsum(chord_fractions)
        self.foil_list = [self.base_foil]
        for i in range(self.n_foils - 1):
            adj_x = np.zeros(2)
            if i > 0:
                adj_x[0] = self.split_lengths[i - 1]

            raw_left_foil, raw_right_foil, ref_indices = airfoil_splitter(self.foil_list[i] - adj_x, self.chord_fractions[i], hinge_positions[i], nose_up_thetas[i], nose_down_thetas[i], og_profile=self.base_foil - adj_x, straight_rear=straight_rear, smoothen_tip=smoothen_tip, N_arc_points=N_arc_points, theta_lim_deg=theta_lim_deg, do_ref_split=self.do_ref_split)
            left_foil = raw_left_foil + adj_x
            right_foil = raw_right_foil + adj_x

            # Make foil profiles immutable
            left_foil.flags.writeable = False
            right_foil.flags.writeable = False

            self.foil_list[i] = left_foil
            self.foil_list.append(right_foil)

            # Store the ref_split_indices for the rear of the left airfoil if applicable
            if self.do_ref_split:
                self.rear_ref_split_indices.append(ref_indices[0])
                self.front_ref_aux_angles.append(ref_indices[1])

    def rotate(self, angles, save=False, plot=False):
        angles = np.cumsum(angles)  # the user provides relative, not absolute angles

        # The first airfoil never rotates (then simply change the angle of attack)
        if self.do_ref_split:
            local_foil_list = sever_rear(self.foil_list[0], self.rear_ref_split_indices[0])
        else:
            local_foil_list = [self.foil_list[0]]
        hinge_list = []
        rotated_foil = self.foil_list[0]
        for i in range(self.n_foils - 1):
            og_hinge_pos = np.array([self.split_lengths[i], 0]) + self.hinge_positions[i]

            if i == 0:
                # First hinge does not move
                new_hinge_pos = og_hinge_pos
            else:
                # The other hinge positions are based on the previous ones
                new_hinge_pos = new_hinge_pos + rotate(np.array([self.chord_fractions[i], 0] + self.hinge_positions[i] - self.hinge_positions[i - 1]), angles[i - 1])

            hinge_list.append(new_hinge_pos)
            new_rotated_foil = rotate(self.foil_list[i + 1] - og_hinge_pos, angles[i]) + new_hinge_pos
            if self.do_ref_split:
                if i == self.n_foils - 2:
                    # last airfoil does not have a rear split
                    mid = new_rotated_foil
                    rear = None
                else:
                    mid, rear = sever_rear(new_rotated_foil, self.rear_ref_split_indices[i + 1])

                # We set the size of the front refinement section depending on the current hinge deflection:
                # First find the angular description of the to-be-severed range
                top_point_left_angle = get_2pi_vec_angle(new_hinge_pos, rotated_foil[self.rear_ref_split_indices[i][0]])
                top_point_right_angle = self.front_ref_aux_angles[i][0] + angles[i]
                bot_point_left_angle = get_2pi_vec_angle(new_hinge_pos, rotated_foil[self.rear_ref_split_indices[i][1]])
                bot_point_right_angle = self.front_ref_aux_angles[i][1] + angles[i]

                lim_top = top_point_left_angle - (top_point_right_angle - top_point_left_angle)
                lim_bot = bot_point_left_angle + (bot_point_left_angle - bot_point_right_angle)

                # Then compute the point indices along mid and assemble the mid and front profile parts
                point_angles = get_2pi_vec_angle(new_hinge_pos, mid)
                front_mask = (lim_top <= point_angles) & (point_angles <= lim_bot)
                new_mid = mid[np.invert(front_mask)]

                front_mask[np.argwhere(front_mask)[0] - 1] = True  # make sure to widen the front mask by one in order to ensure point overlap
                front_mask[np.argwhere(front_mask)[-1] + 1] = True
                front = mid[front_mask]
                front = np.vstack((front, front[0]))  # close the front contour
                local_foil_list.append(front)
                local_foil_list.append(new_mid)
                if rear is not None:
                    local_foil_list.append(rear)

            else:
                local_foil_list.append(new_rotated_foil)

            rotated_foil = new_rotated_foil

        if save:
            for i in range(len(local_foil_list)):
                np.savetxt(self.base_name + f'_{i}.dat', local_foil_list[i])

        if plot:
            hinges = self.hinge_positions
            hinges[:, 0] += self.split_lengths
            airfoil_plotter(local_foil_list, hinges=np.array(hinge_list))
