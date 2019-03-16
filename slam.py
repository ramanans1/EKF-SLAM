from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
#https://ac.els-cdn.com/S0921889002002336/1-s2.0-S0921889002002336-main.pdf?_tid=d35bc9db-3c25-49c5-9f79-a65836027bc7&acdnat=1551829154_6ce461d1c7c7f5a9b00a8a4351b0fa41

def motion_model(u, dt, ekf_state, vehicle_params):
    '''
    Computes the discretized motion model for the given vehicle as well as its Jacobian

    Returns:
        f(x,u), a 3x1 vector corresponding to motion x_{t+1} - x_t given the odometry u.

        df/dX, the 3x3 Jacobian of f with respect to the vehicle state (x, y, phi)
    '''

    v_e = u[0]
    alpha = u[1].copy()
    v_c = (v_e)/(1-np.tan(alpha)*(vehicle_params['H']/vehicle_params['L']))
    t_st = ekf_state['x'].copy()
    phi = t_st[2]
    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']
    el1 = dt*(v_c*np.cos(phi)-(v_c/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    el2 = dt*(v_c*np.sin(phi)+(v_c/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    el3 = dt*(v_c/L)*np.tan(alpha)
    el31 = slam_utils.clamp_angle(el3)
    motion = np.array([[el1],[el2],[el31]])
    el13 = -dt*v_c*(np.sin(phi)+(1/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    el23 = dt*v_c*(np.cos(phi)-(1/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    G = np.array([[1,0,el13],[0,1,el23],[0,0,1]])

    #print(motion)
    #print(G)

    return motion, G

def odom_predict(u, dt, ekf_state, vehicle_params, sigmas):
    '''
    Perform the propagation step of the EKF filter given an odometry measurement u
    and time step dt where u = (ve, alpha) as shown in the vehicle/motion model.

    Returns the new ekf_state.
    '''
    t_st = ekf_state['x'].copy()
    t_st = np.reshape(t_st,(t_st.shape[0],1))
    t_cov = ekf_state['P'].copy()
    dim = t_st.shape[0]-3
    F_x = np.hstack((np.eye(3),np.zeros((3,dim))))
    mot, g = motion_model(u,dt,ekf_state,vehicle_params)
    new_x = t_st + np.matmul(np.transpose(F_x),mot)

    R_t = np.zeros((3,3))
    R_t[0,0] = sigmas['xy']*sigmas['xy']
    R_t[1,1] = sigmas['xy']*sigmas['xy']
    R_t[2,2] = sigmas['phi']*sigmas['phi']
    Gt_1 = np.hstack((g, np.zeros((3,dim))))
    Gt_2 = np.hstack((np.zeros((dim,3)),np.eye(dim)))
    Gt = np.vstack((Gt_1,Gt_2))
    new_cov = np.matmul(Gt,np.matmul(t_cov,np.transpose(Gt)))+np.matmul(np.transpose(F_x),np.matmul(R_t,F_x))
    new_cov = slam_utils.make_symmetric(new_cov)
    new_x = np.reshape(new_x,(new_x.shape[0],))
    ekf_state['x'] = new_x
    ekf_state['P'] = new_cov

    return ekf_state


def gps_update(gps, ekf_state, sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x,y).

    Returns the updated ekf_state.
    '''

    ###
    # Implement the GPS update.
    ###
    print('GPSP')
    P = ekf_state['P']
    P_mat = np.matrix(P)
    r = np.transpose([gps - ekf_state['x'][:2]])
    H_mat = np.matrix(np.zeros([2, P.shape[0]]))
    H_mat[0, 0] = 1
    H_mat[1, 1] = 1
    R_mat = np.matrix(np.zeros([2, 2]))
    R_mat[0, 0] = sigmas['gps'] ** 2
    R_mat[1, 1] = sigmas['gps'] ** 2
    S_mat = H_mat * P_mat * H_mat.T + R_mat
    d = (np.matrix(r)).T * np.matrix(slam_utils.invert_2x2_matrix(np.array(S_mat))) * np.matrix(r)
    if d <= chi2.ppf(0.999, 2):
        K_mat = P_mat * H_mat.T * np.matrix(slam_utils.invert_2x2_matrix(np.array(S_mat)))
        ekf_state['x'] = ekf_state['x'] + np.squeeze(np.array(K_mat * np.matrix(r)))
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        ekf_state['P'] = slam_utils.make_symmetric(
            np.array((np.matrix(np.eye(P.shape[0])) - K_mat * H_mat) * P_mat))

    return ekf_state

def laser_measurement_model(ekf_state, landmark_id):
    '''
    Returns the measurement model for a (range,bearing) sensor observing the
    mapped landmark with id 'landmark_id' along with its jacobian.

    Returns:
        h(x, l_id): the 2x1 predicted measurement vector [r_hat, theta_hat].

        dh/dX: For a measurement state with m mapped landmarks, i.e. a state vector of
                dimension 3 + 2*m, this should return the full 2 by 3+2m Jacobian
                matrix corresponding to a measurement of the landmark_id'th feature.
    '''

    ###
    # Implement the measurement model and its Jacobian you derived
    ###

    return zhat, H

def initialize_landmark(ekf_state, tree):
    '''
    Initialize a newly observed landmark in the filter state, increasing its
    dimension by 2.

    Returns the new ekf_state.
    '''

    ###
    # Implement this function.
    ###

    return ekf_state

def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###

    return assoc

def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function initialize_landmark
    in the state for measurement i. If assoc[i] == -2, discard the measurement as
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }

    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.init_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = gps_update(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = odom_predict(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.do_plot(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params)


        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.read_data_file("data/DRS.txt")
    gps = slam_utils.read_data_file("data/GPS.txt")
    laser = slam_utils.read_data_file("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50,
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
