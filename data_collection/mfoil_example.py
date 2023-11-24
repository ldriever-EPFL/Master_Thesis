from mfoil import MFoil


# # --------------- MFoil Usage Example --------------- # #
"""
Split the NACA0012 airfoil into sections, rotate the sections, and display and save the result
"""

chord_fractions = [0.65, 0.2]  # Define where along the chord the airfoil is split
hinge_positions = [[0.06, 0.01], [0.025, 0]]  # Hinge positions defining the center of the 'splitting arc'
nose_up_thetas = [0.5, 0.6]  # Maximum nose up deflections in radians
nose_down_thetas = [0.5, 0.6]  # Maximum nose down deflections in radians
foil = MFoil('naca0012.dat', chord_fractions, hinge_positions, nose_up_thetas, nose_down_thetas)

rotation_angles = [0.2, -0.3]
foil.rotate(rotation_angles, plot=True, save=True)

