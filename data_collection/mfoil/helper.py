import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def airfoil_plotter(foil_list, hinges=None):
    for i in range(len(foil_list)):
        profile = foil_list[i]

        plt.plot(profile[:, 0], profile[:, 1], linestyle='-')

        if hinges is not None and i < len(hinges):
            plt.plot(hinges[i, 0], hinges[i, 1], marker='o', color='black')

    plt.axis('equal')
    plt.show()
    

def compute_intersection_shape(curve_1, curve_2):
    # Create the Polygon objects and find their intersection(s)
    polygon1 = Polygon(curve_1)
    polygon2 = Polygon(curve_2)
    intersection_all = polygon1.intersection(polygon2)

    # Check if there is a single polygon result (innermost intersection)
    if intersection_all.geom_type == 'Polygon':
        innermost_intersection = intersection_all
    else:
        # If there are multiple polygons, find the one with the smallest area
        innermost_intersection = min(intersection_all, key=lambda x: x.area)

    # Extract and return the coordinates of the innermost intersection
    intersection_coords = list(innermost_intersection.exterior.coords)
    return np.array(intersection_coords)


def sort_cc(points):
    # Remove any duplicates
    points = np.unique(points, axis=0)

    # Sort the points in cc order
    min_x = np.mean(points[:, 0]) + 0.001
    angles = np.arctan2(points[:, 1], points[:, 0] - min_x)
    angles[angles < 0] += 2 * np.pi
    sorted_points = points[np.argsort(angles)]

    # First point must be the last
    return np.vstack((sorted_points, sorted_points[0]))


def get_2pi_vec_angle(p1, p2):
    """
    Finds the angle of the vector p2 - p1
    :param p1: must have two entries
    :param p2: can be nx2
    :return:
    """
    if len(p2.shape) == 1:
        # Turn into column vector
        p2 = p2.reshape(1, 2)

    vec = p2 - p1
    angles = np.arctan2(vec[:, 1], vec[:, 0])
    angles = np.mod(angles, 2 * np.pi)

    if p2.size == 2:
        angles = angles[0]

    return angles


def rotate(vec, theta):
    """
    :param vec: the to-be rotated vector (array of shape nx2)
    :param theta: rotation angle in radians
    :return: the rotated vector (nx2)
    """
    if len(vec.shape) == 1:
        # Turn into column vector
        vec = vec.reshape(1, 2)

    if vec.shape[1] != 2:
        raise Exception("Input vector must be of shape 2xn")

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    if np.size(theta) == 1:
        rot_vec = np.einsum('ij,kj->ki', rot_matrix, vec)
    else:
        rot_vec = np.einsum('ijk,kj->ki', rot_matrix, vec)

    rot_vec[abs(rot_vec) < 1e-15] = 0  # remove numerical precision errors

    if vec.size == 2:
        rot_vec = rot_vec[0]

    return rot_vec


def find_lines_intersection(p1, p2, d1, d2):
    if np.any(np.abs(d1[0] / d2[0] - d1[1] / d2[1]) < 0.001 * np.abs(d1[0] / d2[0])):
        raise Exception('lines are almost parallel')

    r1 = (p2[1] - p1[1] + d2[1] / d2[0] * (p1[0] - p2[0])) / (d1[1] - d1[0] * d2[1] / d2[0])
    center = p1 + r1 * d1

    return center


def find_lines_angle(d1, d2):
    return np.arccos(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))


def compute_smooth_points(tbr_points, N):
    # tbr_points are the cc-sorted points which are to be replaced by a curve with N points

    if len(tbr_points) < 3:
        raise Exception("tbr_points must have length of at least 3")
    
    p1 = tbr_points[0]
    p2 = tbr_points[-1]
    t1 = tbr_points[0] - tbr_points[1]
    t2 = tbr_points[-1] - tbr_points[-2]

    n1 = np.array([t1[1], -t1[0]])
    n2 = np.array([t2[1], -t2[0]])

    circle_center = find_lines_intersection(p1, p2, n1, n2)

    r1 = np.linalg.norm(circle_center - p1)
    r2 = np.linalg.norm(circle_center - p2)
    circ_angle = find_lines_angle(p1 - circle_center, p2 - circle_center)

    if r2 > r1:
        # take p2 as fixed
        r = (r1 - r2 * np.cos(circ_angle)) / (1 - np.cos(circ_angle))
        new_center = p2 - (p2 - circle_center) * r / r2

        end_angle = np.arctan2((p2 - circle_center)[1], (p2 - circle_center)[0])
        start_angle = end_angle - circ_angle
        angles = np.linspace(start_angle, end_angle, N-1)
        arc_points = new_center + r * np.hstack((np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)))
        direction = (arc_points[0] - p1)
        arc = np.vstack(((p1 + np.linspace(0, 1, max(N, int(N * np.linalg.norm(direction) * 10))).reshape(-1, 1) * direction)[:-1], arc_points))

    else:
        # take p1 as fixed
        r = (r2 - r1 * np.cos(circ_angle)) / (1 - np.cos(circ_angle))
        new_center = p1 - (p1 - circle_center) * r / r1

        start_angle = np.arctan2((p1 - circle_center)[1], (p1 - circle_center)[0])
        end_angle = start_angle + circ_angle
        angles = np.linspace(start_angle, end_angle, N - 1)
        arc_points = new_center + r * np.hstack((np.cos(angles).reshape(-1, 1), np.sin(angles).reshape(-1, 1)))
        direction = (p2 - arc_points[-1])
        arc = np.vstack((arc_points, (p2 - np.linspace(0, 1, max(N, int(N * np.linalg.norm(direction) * 10))).reshape(-1, 1) * direction)[:-1]))

    return arc


def detect_kinks(points, theta_lim_deg=5):
    theta_lim = theta_lim_deg / 180 * np.pi
    local_tangent_vecs = points[1:] - points[:-1]
    local_tangent_angles = np.arctan2(local_tangent_vecs[:, 1], local_tangent_vecs[:, 0])
    local_tangent_angles[local_tangent_angles < 0] += 2 * np.pi

    tangent_angle_diffs = local_tangent_angles[1:] - local_tangent_angles[:-1]

    kink_indices = np.ravel(np.argwhere(np.abs(tangent_angle_diffs) > theta_lim)) + 1  # plus one to get the index of the point at the center of the kink
    return kink_indices


def smoothen_profile_large(points, theta_lim_deg, N_arc=100):

    # find the indices describing unique kinks
    kink_indices = detect_kinks(points, theta_lim_deg)
    lst = []
    last_index = -1e10
    for idx in kink_indices:
        if idx > last_index + 10:  # Don't smooth overlapping sections
            last_index = idx
            lst.append(idx)

    if len(lst) > 4 or len(lst) == 1:
        raise Exception(f"{len(lst)} kinks detected. Adjust theta_lim")

    if len(lst) == 0:
        # No kinks to fix
        return points

    if len(lst) == 4:
        lst = lst[1:3]

    if len(lst) == 3:
        lst = [lst[1], lst[1]]

    # now handle the kinks
    pts_up = compute_smooth_points(points[:lst[0]], N_arc)
    pts_down = compute_smooth_points(points[lst[1] + 1:], N_arc)

    new_points = sort_cc(np.vstack((pts_up, pts_down, points[lst[0]: lst[1] + 1])))

    return sort_cc(new_points)


def smoothen_profile_detail(points, theta_lim_deg, N_smooth, N_arc=None):
    if N_arc is None:
        N_arc = 4 * N_smooth + 1  # two points per original point

    kink_indices = detect_kinks(points, theta_lim_deg)
    kink_mask = np.ones(len(points), dtype=bool)

    # find the indices describing unique kinks
    new_points = None
    last_index = -1e10
    for idx in kink_indices:
        if idx > last_index + 2 * N_smooth:  # Don't smooth overlapping sections
            last_index = idx
            if not points[idx, 0] == max(points[:, 0]):
                kink_mask[idx - N_smooth : idx + N_smooth + 1] = False
                new_local_points = compute_smooth_points(points[max(0, idx - N_smooth) : idx + N_smooth + 1], N_arc)
                if new_points is None:
                    new_points = new_local_points
                else:
                    new_points = np.vstack((new_points, new_local_points))

    new_points = np.vstack((points[kink_mask], new_points))

    return sort_cc(new_points)


def airfoil_splitter(profile, split_length, hinge_position, theta_max_up, theta_max_down, og_profile, straight_rear=True, N_arc_points=100, N_smooth=10, theta_lim_deg=None, N_smooth_arc=None, smoothen_tip=True, do_ref_split=False):
    """
    :param N_arc_points: number of points used to discretize the splitting semi-circle. Not the number of points ultimately present in the airfoil
    :return: saves new profiles as files
    """

    # Read and set up the initial parameters
    hinge_position = np.array(hinge_position)
    absolute_hinge_pos = hinge_position + np.array([split_length, 0])
    N_smooth += 3

    theta_max_up = np.abs(theta_max_up)  # to avoid confusion in the definition of these parameters
    theta_max_down = np.abs(theta_max_down)  # to avoid confusion in the definition of these parameters

    hinge_radius = np.linalg.norm(hinge_position)

    if do_ref_split and (straight_rear or smoothen_tip):
        raise Exception("finding the rear ref_split indices for the left airfoil is only implemented for straight_Rear=False and smoothen_tip=False")

    # Use arc to split off right airfoil
    split_arc_thetas = np.linspace(np.pi / 2, 3 * np.pi / 2, N_arc_points).reshape(-1, 1)
    split_arc = np.vstack(([5, 5], np.hstack((np.cos(split_arc_thetas), np.sin(split_arc_thetas))) * hinge_radius + absolute_hinge_pos, [5, -5]))
    right_profile = compute_intersection_shape(profile, split_arc)

    # Account for max nose-up deflection
    cutting_profile_top = np.vstack(([2, 2], rotate(og_profile[og_profile[:, 1] > 0] - absolute_hinge_pos, theta_max_up) + absolute_hinge_pos, [3, -3]))
    right_profile = compute_intersection_shape(right_profile, cutting_profile_top)

    # Account for max nose-down deflection
    cutting_profile_bottom = np.vstack(([-2, 2], rotate(og_profile[og_profile[:, 1] < 0] - absolute_hinge_pos, -theta_max_down) + absolute_hinge_pos, [3, 3]))
    right_profile = sort_cc(compute_intersection_shape(right_profile, cutting_profile_bottom))

    # Smoothen right profile
    right_profile = smoothen_profile_large(right_profile, theta_lim_deg, N_smooth)
    if smoothen_tip:
        right_profile = smoothen_profile_detail(right_profile, theta_lim_deg, 5, N_arc=N_smooth_arc)

    # Complete the left profile:
    left_ref_indices = None
    right_ref_angles = None
    if straight_rear:
        x_lim = min(right_profile[:, 0])
        left_profile = sort_cc(compute_intersection_shape(profile, [[x_lim, 2], [-2, 2], [-2, -2], [x_lim, -2]]))
    else:
        left_profile = sort_cc(compute_intersection_shape(profile, np.vstack((split_arc[1:-1], [20, -20], [-20, -20], [-20, 20], [20, 20]))))

        if do_ref_split:
            # find which part of the left profile are a result of the arc-intersection
            left_profile_in_ref_section_indices = np.array(sorted(set(np.where(np.all(left_profile[:, np.newaxis, :] == split_arc[1:-1][np.newaxis, :, :], axis=2))[0])))
            # points will start from roughly in the middle of the arc (not exactly the middle if there is a hinge offset) so we need to deal with the arc in two parts
            arc_top_last_idx = left_profile_in_ref_section_indices[np.argwhere(np.diff(left_profile_in_ref_section_indices) > 1)[0]] + 1
            arc_bot_first_idx = left_profile_in_ref_section_indices[np.argwhere(np.diff(left_profile_in_ref_section_indices) > 1)[0] + 1] - 1
            left_ref_indices = [arc_top_last_idx[0], arc_bot_first_idx[0] - len(left_profile)]
            
            # Repeat the procedure for the right profile
            right_profile_in_ref_section_indices = np.array(sorted(set(np.where(np.all(right_profile[:, np.newaxis, :] == split_arc[1:-1][np.newaxis, :, :], axis=2))[0])))
            # Find the angles relative to the hinge
            right_ref_angles = [get_2pi_vec_angle(absolute_hinge_pos, right_profile[right_profile_in_ref_section_indices[0]]), get_2pi_vec_angle(absolute_hinge_pos, right_profile[right_profile_in_ref_section_indices[-1]])]

    return left_profile, right_profile, [left_ref_indices, right_ref_angles]


def sever_rear(points, split_indices):
    back = np.vstack((points[split_indices[1] :], points[: split_indices[0] + 1]))
    # Compute a point on the outside of the arc (thus on the inside of the profile) that can be used to close the contour without external effects
    corner = find_lines_intersection(back[0], back[-1], back[1] - back[0], back[-1] - back[-2])
    back = np.vstack((corner, back[np.sort(np.unique(back, return_index=True, axis=0)[1])], corner))
    front = np.vstack((corner, points[split_indices[0] : split_indices[1] + 1], corner))
    return [front, back]


def refine2(points):
    # Double the number of points by splitting each linear section in 2
    tba_points = points[:-1] + 0.5 * (points[1:] - points[:-1])

    return sort_cc(np.vstack((tba_points, points)))


def py_float_from_fortran(number_string):
    return float(number_string.replace('d', 'e'))
