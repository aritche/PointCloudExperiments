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

def fast_vis(trk_cloud, vector_cloud):
    # Get streamlines and seeds from tensors
    #streamlines = np.reshape(streamlines, (-1,3))
    streamlines = trk_cloud 

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

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    positions, angles = vector_cloud[:,:3], vector_cloud[:,3:]
    vis_angles = angles.copy()
    vis_angles[:,0] /= 145
    vis_angles[:,1] /= 174
    vis_angles[:,2] /= 145
    ax.quiver(positions[:,0], positions[:,1], positions[:,2], vis_angles[:,0], vis_angles[:,1], vis_angles[:,2])
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

    # Move data back to the old origin [0.5, 0.5, 0.5]
    result = result + np.array([0.5, 0.5, 0.5])
    
    return result

def scale_trk_cloud(trk_cloud, scale_matrix):
    # Move data to the origin [0,0,0]
    trk_cloud = trk_cloud - np.array([0.5,0.5,0.5])

    result = np.dot(trk_cloud, scale_matrix.T)

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

def scale_tom_cloud(tom_cloud, scale_matrix):
    positions, angles = tom_cloud[:,:3], tom_cloud[:,3:]

    positions = positions - np.array([0.5,0.5,0.5])
    positions = np.dot(positions, scale_matrix.T)
    positions = positions + np.array([0.5, 0.5, 0.5])

    result = np.concatenate((positions,angles), axis=-1)
    return result

def tom_add_noise(tom_cloud, mean, stdev):
    positions, angles = tom_cloud[:,:3], tom_cloud[:,3:]
    noise = np.random.normal(loc=0, scale=stdev, size=angles.shape)
    angles += noise

    tom_cloud = np.concatenate((positions,angles), axis=-1)
    return tom_cloud

def displace_tom_cloud(tom_cloud, x, y, z):
    positions, angles = tom_cloud[:,:3], tom_cloud[:,3:]

    positions += np.array([x,y,z])

    tom_cloud = np.concatenate((positions,angles), axis=-1)

    return tom_cloud

def displace_trk_cloud(trk_cloud, x, y, z):
    return trk_cloud + np.array([x,y,z])


# Rotation factors
x_angle = np.random.uniform(-np.pi/4, np.pi/4)
y_angle = np.random.uniform(-np.pi/4, np.pi/4)
z_angle = np.random.uniform(-np.pi/4, np.pi/4)

# Scale factors
x_factor = np.random.uniform(0.9, 1.5)
y_factor = np.random.uniform(0.9, 1.5)
z_factor = np.random.uniform(0.9, 1.5)

# Displacement factors
x_disp = np.random.uniform(0,0.1)
y_disp = np.random.uniform(0,0.1)
z_disp = np.random.uniform(0,0.1)

# Noise stdev factor
noise_stdev = np.random.uniform(0,0.05)

print('ROTATION:')
print(x_angle, y_angle, z_angle)

print('SCALE:')
print(x_factor, y_factor, z_factor)

print('DISPLACEMENT:')
print(x_disp, y_disp, z_disp)

print('NOISE:')
print(noise_stdev)

rot_matrix = get_rot_matrix(x_angle, y_angle, z_angle)
scale_matrix = get_scale_matrix(x_factor, y_factor, z_factor)

# Open files
trk_fn = '../../data/CST_left_128_10_partial/not_preprocessed/tractogram_clouds/599469_0_CST_left.npy'
tom_fn = '../../data/CST_left_128_10_partial/not_preprocessed/vector_clouds/599469_0_CST_left_DIRECTIONS.npy'
trk_cloud = np.load(trk_fn)
tom_cloud = np.load(tom_fn)

# Augment the TOM cloud
fast_vis(trk_cloud, tom_cloud)
tom_cloud = rotate_tom_cloud(tom_cloud, rot_matrix)
tom_cloud = displace_tom_cloud(tom_cloud, x_disp, y_disp, z_disp)
tom_cloud = scale_tom_cloud(tom_cloud, scale_matrix)
tom_cloud = tom_add_noise(tom_cloud, 0, noise_stdev)

# Augment the TRK cloud
trk_cloud = rotate_trk_cloud(trk_cloud, rot_matrix)
trk_cloud = displace_trk_cloud(trk_cloud, x_disp, y_disp, z_disp)
trk_cloud = scale_trk_cloud(trk_cloud, scale_matrix)
fast_vis(trk_cloud, tom_cloud)
