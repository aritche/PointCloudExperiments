import torch
import nibabel as nib
from nibabel import trackvis
import numpy as np

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

num_points = 10
num_sl = 128

def fast_vis(streamlines, vectors):
    """
    # Get streamlines and seeds from tensors
    streamlines = np.reshape(streamlines, (-1,3))

    # Plot the result
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    ax.scatter(streamlines[:,0], streamlines[:,1], streamlines[:,2])

    # Render the result
    fig.tight_layout()
    plt.show()
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    positions = vectors[:,:3]
    ax.scatter(positions[:,0], positions[:,1], positions[:,2])
    plt.show()

def get_rot_matrix(x_angle, y_angle, z_angle):
    rot_x = np.array([
         [1, 0,        0],
         [0, np.cos(x_angle), -np.sin(x_angle)],
         [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    rot_y = np.array([
         [np.cos(y_angle), 0, np.sin(y_angle)],
         [0, 1, 0],
         [-np.sin(y_angle), 0, np.cos(y_angle)]
    ])

    rot_z = np.array([
         [np.cos(z_angle), -np.sin(z_angle), 0],
         [np.sin(z_angle), np.cos(z_angle), 0],
         [0, 0, 1]
    ])

    rot_matrix = np.matmul(np.matmul(rot_x, rot_y), rot_z)

    return rot_matrix

def get_scale_matrix(x_factor, y_factor, z_factor):
    scale_matrix = np.array([
        [x_factor,0,0],
        [0,y_factor,0],
        [0,0,z_factor],
    ])

    return scale_matrix

    

def rotate_trk_cloud(trk_cloud, rot_matrix):
    # Move data to the origin [0,0,0]
    trk_cloud = trk_cloud - np.array([0.5,0.5,0.5])

    result = np.dot(trk_cloud, rot_matrix.T)
    #result = np.dot(trk_cloud, rot_matrix)
    #print(np.sum(result - resultB))

    # Move data back to the old origin [0.5, 0.5, 0.5]
    result = result + np.array([0.5, 0.5, 0.5])
    
    return result


def rotate_tom_cloud(tom_cloud, rot_matrix):
    positions, angles = tom_cloud[:,:3], tom_cloud[:,3:]

    positions = positions - np.array([0.5,0.5,0.5])
    positions = np.dot(positions, rot_matrix.T)
    positions = positions + np.array([0.5, 0.5, 0.5])

    result = np.concatenate((positions,angles), axis=-1)
    return result

def add_noise(tom_cloud, mean, stdev):
    positions, angles = tom_cloud[:,:3], tom_cloud[:,3:]
    noise = np.random.normal(loc=0, scale=stdev, size=tom_cloud.shape)
    tom_cloud += noise
    


x_angle, y_angle, z_angle = 0,0,np.pi
rot_matrix = get_rot_matrix(x_angle, y_angle, z_angle)

fn = '../../data/CST_left_128_10_partial/not_preprocessed/tractogram_clouds/599469_0_CST_left.npy'
data = np.load(fn)
result_trk = rotate_trk_cloud(data, rot_matrix)

fn = '../../data/CST_left_128_10_partial/not_preprocessed/vector_clouds/599469_0_CST_left_DIRECTIONS.npy'
tom = np.load(fn)

x_factor, y_factor, z_factor = 0.5, 0.5, 0.5
scale_matrix = get_scale_matrix(x_factor, y_factor, z_factor)
result_tom = rotate_tom_cloud(tom, scale_matrix)

fast_vis(result_tom, result_tom)
