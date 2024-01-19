import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from .quat_tools import *
import random


def _interp_index_list(q_list, index_list, interp=True):
    L = len(index_list)

    # if interp=False, then no interpolation is needed, just need to 
    # make sure each traj starting at 0 and ending at N
    index_list_interp = []

    if interp == True:
        for l in range(L):
            if l != L-1:
                N = index_list[l+1][0] - index_list[l][0] 
            else:
                N = len(q_list) - index_list[l][0]
            index_list_interp.append(np.arange(0, N))

    elif interp == False:
        for l in range(L):
            index_list_interp.append(index_list[l] - index_list[l][0])


            # if l != L-1:
            #     N = index_list[l+1][0] - index_list[l][0] 
            # else:
            #     N = len(q_list) - index_list[l][0]
            # index_list_interp.append(np.arange(0, N))

    # for l in range(L):
    # #     index_list_interp[l] =np.linspace(0, N, num=index_list[l].shape[0], endpoint=True, dtype=int)
    #     index_list_interp.append(index_list[l] - index_list[l][0])
        
    index_list_interp = np.hstack(index_list_interp)


    # find the longest traj as reference

    # Init it starting from 0 to N

    # For the remaining traj, starting from the first point as index 0
    # then compare the rest of points with the reference, and use the index of
    # the closest point from the reference as the new index


    # The last point of the remaining traj should always be the same as N


    # Return the new index list after interpolation
    return index_list_interp


def _plot_rotated_axes(ax, r , offset=(0, 0, 0), scale=1):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])

    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                      colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)

        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)

        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
    
    


def plot_rotated_axes_sequence(q_list, N=3):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    
    seq = np.linspace(0, len(q_list)-1, N, dtype=int)
    for i in range(N):
        _plot_rotated_axes(ax, q_list[seq[i]],  offset=(3*i, 0, 0))


    ax.set(xlim=(-1.25, 1.25 + 3*N-3), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    ax.set(xticks=range(-1, 2 + 3*N-3), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.figure.set_size_inches(2*N, 5)
    # plt.tight_layout()
    # plt.show()





def animate_rotated_axes(R_list, scale=1, **argv):
    """
    List of Rotation object
    """



    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")


    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)


    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    lines = [ax.plot([], [], [], c=c)[0] for c in colors]
    
    if "att" in argv:
        lines_att = [ax.plot([], [], [], c=c)[0] for c in colors]


    def _init():
        for line in lines:
            line.set_data_3d([], [], [])
        
        if "att" in argv:
            for line in lines_att:
                line.set_data_3d([], [], [])


    def _animate(i):
        r = R_list[i]

        if "att" in argv:
            att = argv["att"]
            for axis, (line, c) in enumerate(zip(lines_att, colors)):
                line_ = np.zeros((2, 3))
                line_[1, axis] = scale
                line_rot_ = att.apply(line_)
                line.set_data_3d([line_rot_[0, 0], line_rot_[1, 0]], [line_rot_[0, 1], line_rot_[1, 1]], [line_rot_[0, 2], line_rot_[1, 2]])

        for axis, (line, c) in enumerate(zip(lines, colors)):
            line_ = np.zeros((2, 3))
            line_[1, axis] = scale
            line_rot_ = r.apply(line_)
            line.set_data_3d([line_rot_[0, 0], line_rot_[1, 0]], [line_rot_[0, 1], line_rot_[1, 1]], [line_rot_[0, 2], line_rot_[1, 2]])


        fig.canvas.draw()
        ax.set_title(f'Frame: {i}')


    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                                frames=len(R_list), interval=1000/len(R_list), blit=False, repeat=True)
    
    
    plt.tight_layout()
    plt.show()


def plot_gmm(q_list, index_list, label, interp):

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)

    index_list_interp = _interp_index_list(q_list, index_list, interp)

    color_mapping = np.take(colors, label)

    q_list_q = list_to_arr(q_list)
    for k in range(4):
        ax.scatter(index_list_interp, q_list_q[:, k], s=1, c=color_mapping)

    ax.set_title("GMM results")
    pass


def plot_demo(q_list, index_list, interp, **argv):
    """
    Plot scatter quaternions from demonstrations.
    """
    label_list = ['x', 'y', 'z', 'w']
    colors = ['red', 'blue', 'lime', 'magenta']

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)

    index_list_interp = _interp_index_list(q_list, index_list, interp)

    q_list_q = list_to_arr(q_list)
    for k in range(4):
        ax.scatter(index_list_interp, q_list_q[:, k], s= 1, color=colors[k])

    if "title" in argv:
        ax.set_title(argv["title"])

    pass




def plot_quat(q_list, **argv):

    if "ax" not in argv:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.figure.set_size_inches(12, 6)
    else:
        ax = argv["ax"]

    q_list_q = list_to_arr(q_list)


    label_list = ['x', 'y', 'z', 'w']
    N = q_list_q.shape[0]


    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(4):
        ax.plot(np.arange(N), q_list_q[:, k], color=colors[k], label = label_list[k])

    ax.legend()
    if "title" in argv:
        ax.set_title(argv["title"])

    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    N = q_list_q.shape[0]
    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(4):
        axs[k].plot(np.arange(N), q_list_q[:, k], color=colors[k], label = label_list[k])
        axs[k].legend(loc="upper left")
   
    if "title" in argv:
            axs[0].set_title(argv["title"])
    """
    # plt.show()

    return ax


def plot_4d_coord(q_list, **argv):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)

    label_list = ['x', 'y', 'z', 'w']
    N = q_list.shape[0]


    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(4):
        ax.plot(np.arange(N), q_list[:, k], color=colors[k], label = label_list[k])
    
    if "title" in argv:
        ax.set_title(argv["title"])

    ax.legend()

    return ax



def plot_rot_vec(w_list, **argv):
    
    N = len(w_list)
    w_arr = np.zeros((N, 3))

    for i in range(N):
        # w_arr[i, :] = w_list[i].as_euler('xyz')
        w_arr[i, :] = w_list[i].as_rotvec()


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.figure.set_size_inches(12, 6)


    label_list = ['w_x', 'w_y', 'w_z']

    colors = ['red', 'blue', 'lime', 'magenta']
    for k in range(3):
        ax.plot(np.arange(N), w_arr[:, k], color=colors[k], label = label_list[k])

    
    if "title" in argv:
        ax.set_title(argv["title"])
        
    ax.legend()

    return ax



def plot_gmm_prob(w_arr, **argv):

    N, K = w_arr.shape

    fig, axs = plt.subplots(K, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(K):
        axs[k].scatter(np.arange(N), w_arr[:, k], s=5, color=colors[k])
        axs[k].set_ylim([0, 1])
    
    if "title" in argv:
        axs[0].set_title(argv["title"])






if __name__ == "__main__":

    
    r0 = R.identity()
    r1 = R.from_euler("ZYX", [90, -30, 0], degrees=True)  # intrinsic
    r2 = R.from_euler("zyx", [90, -30, 0], degrees=True)  # extrinsic
    r_list = [r0, r1, r2]



    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d", proj_type="ortho")
    # # ax.figure.set_size_inches(10, 8)
    # # ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    # # ax.set_aspect("equal", adjustable="box")
    # # plot_sequence_rotated_axes(ax, r_list)

    # ax.set(xlim=(-1.25, 7.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
    # ax.set(xticks=range(-1, 8), yticks=[-1, 0, 1], zticks=[-1, 0, 1])
    # ax.set_aspect("equal", adjustable="box")
    # ax.figure.set_size_inches(6, 5)
    # plt.tight_layout()
    # plt.show()

    plot_rotated_axes_sequence(r_list)

    # animate_rotated_axes(ax, r_list)

    # plot_rotated_axes(ax, r1)

