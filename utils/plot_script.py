import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints1, joints2, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = np.minimum(data1.min(axis=0).min(axis=0), data2.min(axis=0).min(axis=0))
    MAXS = np.maximum(data1.max(axis=0).max(axis=0), data2.max(axis=0).max(axis=0))
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data1.shape[0]
    #     print(data.shape)

    height_offset1 = (data1.min(axis=0).min(axis=0))[1]
    height_offset2 = (data2.min(axis=0).min(axis=0))[1]
    data1[:, :, 1] -= height_offset1
    data2[:, :, 1] -= height_offset2
    trajec1 = data1[:, 0, [0, 2]]
    trajec2 = data2[:, 0, [0, 2]]
    
    #     print(trajec.shape)
    lines=[]
    def update(index):
        #         print(index)
        
        for line in lines:
            line[0].remove()
        lines.clear()
        collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        
        
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)



        for i, chain in enumerate(kinematic_tree):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            
            line = ax.plot(data1[index, chain, 0], data1[index, chain, 1], data1[index, chain, 2], linewidth=linewidth,
                      color=colors[0])
            lines.append(line)
            line = ax.plot(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth,
                      color=colors[1])
            lines.append(line)
            
        #         print(trajec[:index, 0].shape)
        
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    ax.set_xlim([MINS[0], MAXS[0]])
    ax.set_ylim([0, MAXS[1]])
    ax.set_zlim([MINS[2], MAXS[2]])
    plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps, writer='ffmpeg')
    plt.close()
