import os
from func_definition import CFD_func
from os.path import join
import numpy as np
from scipy.optimize import minimize


def find_control_cheb_equilibrium(alpha, lfoil, trim_tab_def, cheb_interp, intervals, zero_tol=1e-5):
    """
    Uses the TT Chebyshev interpolation to estimate the equilibeium deflection of the control surface for a specified condition

    :param alpha:           Angle of attack in degrees
    :param lfoil:           Chord fraction of the main control
    :param trim_tab_def:    Deflection of the trim tab in radians
    :param cheb_interp:     Instance of the ChebTT, preChebTT, or ClassicCheb classes
    :param intervals:       Array describing the interpolation intervals, or path to a file containing said array
    :param zero_tol:        Value below which the control hinge Cm is considered to effectively be zero
    :return:                Equilibrium control deflection in radians
    """

    # Load the interval data and create the preChebTT instance
    if type(intervals) == str:
        intervals = np.loadtxt(intervals)
    else:
        intervals = np.array(intervals)

    # Do a constrained 1D minimization of the square of the hinge moment, such that f(Cm) = Cm^2 and df/dx = 2Cm * dCm/dx
    min_object = minimize(lambda x: cheb_interp[alpha, lfoil, x[0], trim_tab_def] ** 2,
                          x0=np.mean(intervals[2]),
                          jac=lambda x: 2 * cheb_interp[alpha, lfoil, x[0], trim_tab_def] * cheb_interp.get_derivative([alpha, lfoil, x[0], trim_tab_def], axis=2),
                          bounds=[intervals[2]])

    if not min_object.success or np.abs(min_object.fun) > zero_tol or min_object.x in intervals[2]:
        # It is very unlikely that the result is exactly on the interval endpoints, so this is more likely a sign of a faulty result
        return np.inf

    return min_object.x[0]


def find_single_control_CFD_equilibrium(alpha, lfoil, trim_tab_def, control_def_min, control_def_max, func, zero_tol=1e-5, max_iter=10, verbose=True, results_directory='sim_outputs'):
    """
    Uses the CFD simulation software to estimate the equilibeium deflection of the control surface for a specified condition

    :param alpha:               Angle of attack in degrees
    :param lfoil:               Chord fraction of the main control
    :param trim_tab_def:        Deflection of the trim tab in radians
    :param control_def_min:     Minimal control surface deflection allowed (in radians)
    :param control_def_max:     Maximal control surface deflection allowed (in radians)
    :param func:                Callable, which executes the
    :param zero_tol:            Value below which the control hinge Cm is considered to effectively be zero
    :param max_iter:            Maximum number of iterations that the line search will do
    :param verbose:             Bool. If False, the simulation software does not print any information to the display
    :param results_directory:   Directory where the full output of the simulation software is stored
    :return:                    Equilibrium control deflection in radians
    """
    
    Cm_theta_min = func(alpha, lfoil, control_def_min, trim_tab_def, new_results_directory=results_directory, existing_results_directory=results_directory, verbose=verbose)
    Cm_theta_max = func(alpha, lfoil, control_def_max, trim_tab_def, new_results_directory=results_directory, existing_results_directory=results_directory, verbose=verbose)

    if Cm_theta_max is None or Cm_theta_min is None:
        return np.inf

    if abs(Cm_theta_min) < zero_tol:
        print('control_def_min is the equilibrium deflection')
        return control_def_min

    if abs(Cm_theta_max) < zero_tol:
        print('control_def_max is the equilibrium deflection')
        return control_def_max

    if Cm_theta_min > 0:
        if Cm_theta_max > 0:
            print('Both initial Cm vals are positive')
            return None
        pos_endpoint_Cm = Cm_theta_min
        pos_endpoint_def = control_def_min
        neg_endpoint_Cm = Cm_theta_max
        neg_endpoint_def = control_def_max

    else:
        if Cm_theta_max < 0:
            print('Both initial Cm vals are negative')
            return None
        pos_endpoint_Cm = Cm_theta_max
        pos_endpoint_def = control_def_max
        neg_endpoint_Cm = Cm_theta_min
        neg_endpoint_def = control_def_min

    deflections = [control_def_min, control_def_max]
    Cm_vals = [Cm_theta_min, Cm_theta_max]

    for i in range(max_iter):
        print(f"Doing iteration {i}")

        if Cm_vals[-1] < 0:
            # Use positive interval endpoint
            slope = (Cm_vals[-1] - pos_endpoint_Cm) / (deflections[-1] - pos_endpoint_def)
        else:
            # Use negative interval endpoint
            slope = (Cm_vals[-1] - neg_endpoint_Cm) / (deflections[-1] - neg_endpoint_def)

        new_deflection = deflections[-1] - Cm_vals[-1] / slope

        new_Cm = func(alpha, lfoil, new_deflection, trim_tab_def, new_results_directory=results_directory, existing_results_directory=results_directory, verbose=verbose)
        if new_Cm is None:
            return np.inf

        deflections.append(new_deflection)
        Cm_vals.append(new_Cm)

        print("new deflection: ", new_deflection)
        print("new Cm: ", new_Cm)

        if abs(new_Cm) < zero_tol:
            print(f"The CFD equilibrium deflection is {new_deflection}, giving a Cm of {new_Cm}")
            return new_deflection

    print('Equilibrium not found')
    return None


def find_control_CFD_equilibria(intervals, func, points, save_directory='', zero_tol=1e-4, max_iter=10, verbose=True):
    """
    Find the CFD-based estimate for the control surface equilibrium deflections for multiple input combinations


    :param intervals:           Intervals across which the CFD function inputs are defined
    :param func:                Callable, which executes the
    :param points:              Input parameter combinations for which the equilibrium control surface deflection is to be found
    :param save_directory:      Directory where the full output of the simulation software is stored
    :param zero_tol:            Value below which the control hinge Cm is considered to effectively be zero
    :param max_iter:            Maximum number of iterations that the line search will do
    :param verbose:             Bool. If False, the simulation software does not print any information to the display
    :return:                    Equilibrium control deflection in radians for the input points
    """

    # Load the intervals
    if type(intervals) == str:
        intervals = np.loadtxt(intervals)
    else:
        intervals = np.array(intervals)

    # Run the tests
    line_search_def = []

    if not os.path.isdir(join(save_directory, 'sim_outputs')):
        os.makedirs(join(save_directory, 'sim_outputs'))

    # We use np.inf as a dummy value as np.savetxt cannot handle None
    for count, point in enumerate(points):
        print('-------------------------------------')
        print(f'Handling Test point {count + 1}')

        line_search_def.append(find_single_control_CFD_equilibrium(point[0], point[1], point[2], intervals[2][0], intervals[2][1], func, zero_tol, max_iter, verbose=verbose, results_directory='sim_outputs'))
        if line_search_def[-1] is None:
            line_search_def[-1] = np.inf

        # Save the results after each iteration (might as well, it's cheap)
        np.savetxt(join(save_directory, 'line_search_deflections.txt'), line_search_def)