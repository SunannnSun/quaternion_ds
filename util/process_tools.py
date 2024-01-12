import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from util import plot_tools, optimize_tools, quat_tools
from util.gmm import gmm as gmm_class
from util.quat_tools import *
from util.plot_tools import *



def _angular_velocities(q1, q2, dt):
    """
    https://mariogc.com/post/angular-velocity-quaternions/
    """
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])
    


def _shift(q_list, index_list):
    """
    Given multiple sequence of trajecotry, take the last point of each trajectory and average out as the attractor,
    then shift each trajectory so that they all end up at a common attractor

    Note: averaging and shifting will somehow flip the signs of all quaternions. Nevertheless, the resulting sequence
    is still smooth and continuous; proceeding operations wrt the attractor; because the attractor is also inverted hence
    no impact on proceeding Riemannian mappings
    """


    L = len(index_list)

    q_att      = [q_list[index_list[l][-1]]  for l in range(L)]
    q_att_quat = [q_att[l].as_quat() for l in range(L)]
    q_att_avg  = R.from_quat(q_att_quat).mean()

    q_shifted = []
    for l in range(L):
        q_diff     = q_att_avg * q_att[l].inv()
        q_shifted += [q_diff * q_list[index] for index in index_list[l]]

    q_shifted = [q * R.identity().inv() for q in q_shifted] # This line can invert the flipped quaternions back to original
    

    return q_shifted, q_att_avg




def _smooth(q_in, q_att, index_list, opt):
    q_new = []

    for l in range(len(index_list)):
        q_in_l    = [q_in[idx] for idx in index_list[l]]

        if opt == "savgol":
            q_in_att  = quat_tools.riem_log(q_att, q_in_l)

            q_new_att = savgol_filter(q_in_att, window_length=80, polyorder=2, axis=0, mode="nearest")

            q_new_arr = quat_tools.riem_exp(q_att, q_new_att)

            q_new     += [R.from_quat(q_new_arr[i, :]) for i in range(q_new_arr.shape[0])]
    
    
        elif opt == "slerp":
            k = 40
            t_list = [0.1*i for i in range(len(q_in_l))]
            
            idx_list  = np.linspace(0, len(q_in_l)-1, num=int(len(q_in_l)/k), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]
            key_rots  = R.from_quat([q_in_l[i].as_quat() for i in idx_list])
            
            slerp = Slerp(key_times, key_rots)

            idx_list  = np.linspace(0, len(q_in_l)-1, num=int(len(q_in_l)), endpoint=True, dtype=int)
            key_times = [t_list[i] for i in idx_list]

            q_interp = slerp(key_times)
            q_new    += [q_interp[i] for i in range(len(q_interp))]


    return q_new


def _filter(q_in, q_att, index_list):

    q_new = []
    q_out = []
    index_list_new = []

    for l in range(len(index_list)):
        if l == 0:
            index_list_l = np.array([0])
        else:
            index_list_l = np.array([index_list_new[l-1][-1]+1]) # Add 1 onto the last point of the previous traj

        q_in_l  = [q_in[idx] for idx in index_list[l]]

        # gmm   = gmm_class(q_in_l, q_att, index_list = index_list[l])
        # label = gmm.begin()

        N = len(q_in_l)

        max_threshold = 0.01

        threshold = max_threshold * np.ones((N, ))
        # threshold[label==gmm.K-1] = np.linspace(max_threshold, 0, num=np.sum(label==gmm.K-1), endpoint=True)
        threshold = np.linspace(max_threshold, 0, num=N, endpoint=True)

        q_new.append(q_in_l[0])
        for i in np.arange(1, N):
            q_curr = q_new[-1]
            q_next = q_in_l[i]
            dis    = q_next * q_curr.inv()
            if np.linalg.norm(dis.as_rotvec()) < threshold[i]:
                pass
            else:
                q_new.append(q_next)
                q_out.append(q_next)
                index_list_l = np.append(index_list_l, index_list_l[-1]+1)
        q_out.append(q_out[-1])
        index_list_new.append(index_list_l)


    return q_new, q_out, index_list_new



def pre_process(q_in_raw, index_list, opt="savgol"):
    
    q_in, q_att = _shift(q_in_raw, index_list)
    # plot_tools.plot_demo(q_in, index_list, interp=False, title="shifted demonstration")

    q_in        = _smooth(q_in, q_att, index_list, opt) 
    # plot_tools.plot_demo(q_in, index_list, interp=False, title='q_smooth')

    q_in, q_out, index_list = _filter(q_in, q_att, index_list)
    plot_tools.plot_demo(q_in, index_list, interp=True, title='q_filter')

    L = len(index_list)

    q_init_list = [q_in[index_list[l][0]] for l in range(L)]
    q_init_quat = [q_init_list[l].as_quat() for l in range(L)]
    q_init      = R.from_quat(q_init_quat).mean()

    q_att_list = [q_in[index_list[l][-1]] for l in range(L)]
    q_att_quat = [q_att_list[l].as_quat() for l in range(L)]
    q_att      = R.from_quat(q_att_quat).mean()


    return q_in, q_out, q_init, q_att, index_list