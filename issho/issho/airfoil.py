import warnings
import os
import numpy as np
from .helper import get_vector_angle


class Airfoil:
    """
    Class containing the fundamental properties of a wing cross-section
    """

    def __init__(self, name, pos_lh, pos_rh, pos_cm, m, J, l_chord=1, profile=None, base_path=''):
        """
        Note on the coordinate system: the leading edge is always placed at the point (0,0)
        :param name: string. Name of the profile (e.g. NACA0012)
        :param pos_lh: [x, y] position of left-hand hinge in the airfoil coordinate system (with scaled chord length)
        :param pos_rh: [x, y] position of right-hand hinge in the airfoil coordinate system (with scaled chord length). If there is only a left hinge, specify None
        :param pos_cm: [x, y] position of center of mass in the airfoil coordinate system (with scaled chord length)
        :param m: mass per unit span length
        :param J: mass moment of inertia per unit span length, calculated about the center of mass
        :param l_chord: airfoil chord length, which is used to scale the loaded contour
        :param profile: nx2 array. Profile points in counter-clockwise order of the closed airfoil contour (first and last points must be the same). Must already be scaled
        """
        self.name = name
        self.pos_lh = np.array(pos_lh)
        if pos_rh is None:
            pos_rh = [l_chord, 0]
        self.pos_rh = np.array(pos_rh)
        self.pos_cm = np.array(pos_cm)
        self.m = m
        self.J = J
        self.l_chord = l_chord
        self.profile = profile

        # Compute auxilliary lengths and angles
        self.hinge_vector = self.pos_rh - self.pos_lh
        self.hinge_cm_vector = self.pos_cm - self.pos_lh

        self.l_h = np.linalg.norm(self.hinge_vector)  # length between the two hinges
        self.l_cm = np.linalg.norm(self.hinge_cm_vector)  # length between the left hinge and the cm
        self.theta_h = get_vector_angle(self.hinge_vector)  # angle from chord line to hinge line (counter-clockwise positive)
        self.theta_cm = get_vector_angle(self.hinge_cm_vector) - self.theta_h  # angle from hinge line to hinge-cm line (counter-clockwise positive)

        # Load and scale the airfoil profile if necessary
        if self.profile is None:
            try:
                self.profile = np.loadtxt(os.path.join(base_path, "airfoil_profiles/" + self.name.lower() + '.dat')) * self.l_chord
            except FileNotFoundError:
                warnings.warn("\nNo matching airfoil profile found! \n- for now the airfoil is treated as a straight line\n- either add the airfoil.dat file in the 'airfoil_profiles' directory or specify the profile points manually\n")

                self.profile = np.array([[self.l_chord, 0], [0, 0], [self.l_chord, 0]])
