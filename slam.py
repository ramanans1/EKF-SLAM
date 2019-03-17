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

    v_e = u[0].copy()
    alpha = u[1].copy()
    H = vehicle_params['H']
    L = vehicle_params['L']
    a = vehicle_params['a']
    b = vehicle_params['b']
    v_c = (v_e)/(1-np.tan(alpha)*(H/L))
    t_st = ekf_state['x'].copy()
    phi = t_st[2]
    el1 = dt*(v_c*np.cos(phi)-(v_c/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    el2 = dt*(v_c*np.sin(phi)+(v_c/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    el3 = dt*(v_c/L)*np.tan(alpha)
    el31 = slam_utils.clamp_angle(el3)
    motion = np.array([[el1],[el2],[el31]])
    el13 = -dt*v_c*(np.sin(phi)+(1/L)*np.tan(alpha)*(a*np.cos(phi)-b*np.sin(phi)))
    el23 = dt*v_c*(np.cos(phi)-(1/L)*np.tan(alpha)*(a*np.sin(phi)+b*np.cos(phi)))
    G = np.array([[1,0,el13],[0,1,el23],[0,0,1]])

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
    P = ekf_state['P']
    dim  = P.shape[0]-2
    H = np.hstack((np.eye(2),np.zeros((2,dim))))
    r = np.transpose([gps - ekf_state['x'][:2]])
    Q = (sigmas['gps']**2)*(np.eye(2))
    S = np.matmul(np.matmul(H,P),H.T) + Q
    S_inv = slam_utils.invert_2x2_matrix(S)
    d = np.matmul(np.matmul(r.T,S_inv),r)
    if d <= chi2.ppf(0.999, 2):
        K = np.matmul(np.matmul(P,H.T),S_inv)
        ekf_state['x'] = ekf_state['x'] + np.squeeze(np.matmul(K,r))
        ekf_state['x'][2] = slam_utils.clamp_angle(ekf_state['x'][2])
        P_temp = np.matmul((np.eye(P.shape[0])- np.matmul(K,H)),P)
        ekf_state['P'] = slam_utils.make_symmetric(P_temp)

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

    t_st = ekf_state['x'].copy()
    t_st[2] = slam_utils.clamp_angle(t_st[2])
    dim = t_st.shape[0]
    r_x,r_y,phi = t_st[0],t_st[1],t_st[2]
    m_x,m_y = t_st[3+2*landmark_id],t_st[4+2*landmark_id]
    del_x = m_x - r_x
    del_y = m_y - r_y
    q = (del_x)**2+(del_y)**2
    sqrt_q = np.sqrt(q)
    zhat = [[sqrt_q],[slam_utils.clamp_angle(np.arctan2(del_y,del_x)-phi)]]
    h = np.array([[-sqrt_q*del_x,-sqrt_q*del_y,0,sqrt_q*del_x,sqrt_q*del_y],[del_y,-del_x,-q,-del_y,del_x]])/q
    F_x = np.zeros((5,dim))
    F_x[:3,:3] = np.eye(3)
    F_x[3,3+2*landmark_id]=1
    F_x[4,4+2*landmark_id]=1
    H = np.matmul(h,F_x)

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
    t_x, t_y, phi = ekf_state['x'][0], ekf_state['x'][1], ekf_state['x'][2]
    m_r, m_th = tree[0], tree[1]

    m_x = t_x + m_r*np.cos(m_th + phi)
    m_y = t_y + m_r*np.sin(m_th + phi)
    ekf_state['x'] = np.hstack((ekf_state['x'], m_x,m_y))

    temp_p = ekf_state['P']
    temp_p = np.vstack((temp_p,np.zeros((1,temp_p.shape[1])),np.zeros((1,temp_p.shape[1]))))
    temp_p = np.hstack((temp_p,np.zeros((temp_p.shape[0],1)),np.zeros((temp_p.shape[0],1))))
    temp_p[temp_p.shape[0]-1,temp_p.shape[0]-1] = 1000
    temp_p[temp_p.shape[0]-2,temp_p.shape[0]-2] = 1000
    ekf_state['P'] = temp_p
    ekf_state['num_landmarks'] = ekf_state['num_landmarks'] + 1

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
        return [-1 for m in measurements]

    n_lmark = ekf_state['num_landmarks']
    n_scans = len(measurements)
    M = np.zeros((n_scans, n_lmark))
    Q_t = np.array([[sigmas['range']**2, 0], [0, sigmas['bearing']**2]])

    alpha = chi2.ppf(0.95, 2)
    beta = chi2.ppf(0.99, 2)
    A = alpha*np.ones((n_scans, n_scans))

    for i in range(n_lmark):
        zhat, H = laser_measurement_model(ekf_state, i)
        S = np.matmul(H, np.matmul(ekf_state['P'],H.T)) + Q_t
        Sinv = slam_utils.invert_2x2_matrix(S)
        for j in range(n_scans):
            temp_z = measurements[j][:2]
            res = temp_z - np.squeeze(zhat)
            M[j, i] = np.matmul(res.T, np.matmul(Sinv, res))

    M_new = np.hstack((M, A))
    pairs = slam_utils.solve_cost_matrix_heuristic(M_new)
    pairs.sort()

    pairs = list(map(lambda x:(x[0],-1) if x[1]>=n_lmark else (x[0],x[1]),pairs))
    assoc = list(map(lambda x:x[1],pairs))

# TODO FIND OUT HOW TO VECTORIZE THIS SECTION, AND THE FOR LOOP ABOVE
    for i in range(len(assoc)):
        if assoc[i] == -1:
            for j in range(M.shape[1]):
                if M[i, j] < beta:
                    assoc[i] = -2
                    break

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
    Q_t = np.array([[sigmas['range']**2, 0], [0, sigmas['bearing']**2]])

    for i in range(len(trees)):
        j = assoc[i]
        if j == -1:
            ekf_state = initialize_landmark(ekf_state, trees[i])
            j = np.int(len(ekf_state['x'])/2) - 2
        elif j == -2:
            continue
        dim = ekf_state['x'].shape[0]
        z_hat, H = laser_measurement_model(ekf_state, j)

        S = np.matmul(H,np.matmul(ekf_state['P'],H.T)) + Q_t
        S_inv = np.linalg.inv(S)
        K = np.matmul(np.matmul(ekf_state['P'],H.T), S_inv)
        z = np.zeros((2, 1))
        z[0,0] = trees[i][0]
        z[1,0] = trees[i][1]

        inno = z - z_hat
        temp_st = ekf_state['x'] + np.squeeze(np.matmul(K, inno))
        temp_st[2] = slam_utils.clamp_angle(temp_st[2])
        ekf_state['x'] = temp_st
        temp_p = np.matmul((np.eye(dim) - np.matmul(K, H)), ekf_state['P'])
        temp_p = slam_utils.make_symmetric(temp_p)
        ekf_state['P'] = temp_p

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
            #print('scan',scan.shape)
            trees = tree_extraction.extract_trees(scan, filter_params)
            #print('trees',trees[0])
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
        "plot_raw_laser": False,
        "plot_map_covariances": False

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
