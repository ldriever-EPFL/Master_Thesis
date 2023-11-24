import numpy as np
from .helper import rotate, get_vector_angle


class FoilBlock:
    """
    Class handling rigidly connected airfoil sections
    """

    def __init__(self, airfoil_list, connection_angles=None):
        """
        :param airfoil_list: list of n Airfoil objects which are to be strung together (through fixed hinges)
        :param connection_angles: list of n - 1 angles (in radians) between the respective airfoil chord lines
        """

        if connection_angles is None:
            connection_angles = []
        self.components = airfoil_list
        self.component_angles = np.ravel([connection_angles])
        connection_angles = np.concatenate((self.component_angles, [0]))
        try:
            self.n = len(airfoil_list)
        except TypeError:
            raise Exception("Input airfoils must be passed as a list, even if it is only one airfoil")

        if self.n == 0:
            raise Exception("At least one airfoil needs to be specified")

        if self.n != len(connection_angles):
            raise Exception("Exactly one less connection angle than the number of airfoils needs to be specified")

        self.hinge_vector = np.array([0, 0])  # Vector from the first (left) hinge to the last (right), given that the first airfoil's chord line is placed along the x-axis
        J_first_hinge = 0  # mass moment of inertia around the first hinge
        self.m = 0  # total mass
        m_times_cm_vec = 0  # vector of first mass moment around the first left hand hinge
        local_cum_angle = 0  # angle between chord line of first airfoil and chord line of current airfoil (i.e. the local cumulative sum of the component angles)

        for count, airfoil in enumerate(self.components):
            first_hinge_new_cm_vector = self.hinge_vector + rotate(airfoil.hinge_cm_vector, local_cum_angle).flatten()
            m_times_cm_vec += first_hinge_new_cm_vector * airfoil.m

            J_first_hinge += airfoil.J + np.linalg.norm(first_hinge_new_cm_vector) ** 2 * airfoil.m  # add the moment of inertia of the new airfoil and use the parallel axis theorem to get the total moment of inertia contribution about the first hinge (J_hinge = J_cm + m_tot * l_cm ** 2)
            self.m += airfoil.m  # simply add the mass of the new airfoil

            self.hinge_vector = self.hinge_vector + rotate(airfoil.hinge_vector, local_cum_angle).flatten()  # update the vector to the currently last hinge
            local_cum_angle += connection_angles[count]

        self.hinge_cm_vector = m_times_cm_vec / self.m  # The center of mass is the first mass moment vector scaled by total mass

        self.l_h = np.linalg.norm(self.hinge_vector)  # length between first and last hinge
        self.l_cm = np.linalg.norm(self.hinge_cm_vector)  # length between the first hinge and the overall cm
        self.theta_h = get_vector_angle(self.hinge_vector)  # angle from chord line of first airfoil to overall hinge line (counter-clockwise positive)
        self.theta_cm = get_vector_angle(self.hinge_cm_vector) - self.theta_h  # angle from overall hinge line to total hinge-cm line (counter-clockwise positive)
        self.first_hinge_angle_diff = self.theta_h - self.components[0].theta_h  # cc-positive angle from hinge line of first airfoil to overall hinge line
        self.last_hinge_angle_diff = self.theta_h - (local_cum_angle + self.components[-1].theta_h)  # cc-positive angle from hinge line of last airfoil to overall hinge line
        self.J = J_first_hinge - self.l_cm ** 2 * self.m  # use the parallel axis theorem to get the moment of inertia about the total body's center of mass (J_cm = J_hinge - m_tot * l_cm ** 2)
