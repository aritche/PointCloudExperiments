"""
A model for generating streamlines for a single tract
Uses 3D TOM input
This is an adaptation of HairNet (Zhou et al. 2018) https://doi.org/10.1007/978-3-030-01252-6_15
"""
import os
import numpy as np
import random
import cv2
import nibabel as nib
from glob import glob

import torch
import torch.nn as nn

from chamfer_python import distChamfer as chamferDistance

from dipy.io.streamline import load_trk
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from nibabel import trackvis
from dipy.tracking import utils

from models.augment import *

num_points = 10
num_streamlines = 128

# Model adapted from https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #self.dropout = torch.nn.Dropout(p=0)
                                                                                                                 # VOLUME SIZE                      # PARAMETERS
        # Encoding (input -> 512 vector)                                                                         # 3 x 144 x 144 x 144 -> 8.9M      (IN * F^3 + 1)*OUT
        self.mlp_1 = nn.Conv1d(in_channels=6,  out_channels=64, kernel_size=1)
        self.mlp_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)

        self.seed_mlp_1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1)
        self.seed_mlp_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)

        self.linear_1 = nn.Linear(in_features=96, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=512)
        self.linear_3 = nn.Linear(in_features=512, out_features=3840)

    def forward(self, tom_cloud, seeds_cloud):
        # Encode TOM
        t = self.relu(self.mlp_1(tom_cloud))
        t = self.relu(self.mlp_2(t))
        t = nn.MaxPool1d(t.size(-1))(t)
        t = t.view(-1, 32)
        #t_encoding = np.reshape(t.cpu().detach().numpy()[0], (1,2048))

        # Encode seeds
        s = (self.relu(self.seed_mlp_1(seeds_cloud)))
        s = (self.relu(self.seed_mlp_2(s)))
        s = nn.MaxPool1d(s.size(-1))(s)
        s = s.view(-1, 64)
        #s_encoding = np.reshape(s.cpu().detach().numpy()[0], (1,64))

        # Concat results
        x = torch.cat((t, s), dim=1)


        #cv2.namedWindow('encoding', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('t_encoding', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('s_encoding', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('linear1', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('linear2', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('linear3', cv2.WINDOW_NORMAL)

        #encoding = np.reshape(x.cpu().detach().numpy()[0], (16,12))
        #encoding = np.reshape(x.cpu().detach().numpy()[0], (192))

        # Generate the final result
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)
        #linear1 = np.reshape(x.cpu().detach().numpy()[0], (32,16))

        #x = self.dropout(self.relu(self.linear_2(x)))
        #linear2 = np.reshape(x.cpu().detach().numpy()[0], (32,32))

        #encoding = (encoding - np.min(encoding))/(np.max(encoding) - np.min(encoding))
        #encoding = (encoding - -1)/(1 - -1)
        #encoding = np.tanh(encoding)
        #cv2.imshow('encoding', np.uint8(encoding*255))


        #t_encoding = (t_encoding - np.min(t_encoding))/(np.max(t_encoding) - np.min(t_encoding))
        #t_encoding = np.tanh(t_encoding)
        #cv2.imshow('t_encoding', np.uint8(t_encoding*255))
        #s_encoding = np.tanh(s_encoding)
        #cv2.imshow('s_encoding', np.uint8(s_encoding*255))
        #encoding = np.tanh(encoding)
        #cv2.imshow('encoding', np.uint8(encoding*255))
        #linear1 = (linear1 - np.min(linear1))/(np.max(linear1) - np.min(linear1))
        #linear1 = (linear1 - -1)/(1 - -1)
        #cv2.imshow('linear1', np.uint8(linear1*255))
        #linear2 = (linear2 - np.min(linear2))/(np.max(linear2) - np.min(linear2))
        #linear2 = (linear2 - -1)/(1 - -1)
        #cv2.imshow('linear2', np.uint8(linear2*255))
        #linear3 = (linear3 - np.min(linear3))/(np.max(linear3) - np.min(linear3))
        #linear3 = (linear3 - -1)/(1 - -1)
        #cv2.imshow('linear3', np.uint8(linear3*255))
        #cv2.waitKey(1)


        #x = self.linear_3(x)

        result = x.view(-1, 3, num_streamlines*num_points)

        return result

# Custom loss function
def CustomLoss(output, target):
    output = output.permute(0,2,1)
    target = target.permute(0,2,1)

    output = output.reshape(-1, num_streamlines, num_points*3)
    target = target.reshape(-1, num_streamlines, num_points*3)

    distA, distB, _, _ = chamferDistance(output, target)

    return (distA + distB).mean()

def get_data(tom_fn, tractogram_fn, is_test):
    # Load data
    tom_cloud = np.load(tom_fn)
    trk_cloud = np.float32(np.load(tractogram_fn))

    # Sample streamlines from tractogram
    trk_cloud = np.reshape(trk_cloud, (-1, num_points*3))
    np.random.shuffle(trk_cloud)
    trk_cloud = trk_cloud[:num_streamlines,:]
    if len(trk_cloud) < num_streamlines: # pad with zeros if not enough streamlines
        padding_cloud = np.zeros((num_streamlines,3*num_points))
        padding_cloud[:trk_cloud.shape[0],:trk_cloud.shape[1]] = trk_cloud
        trk_cloud = padding_cloud
    trk_cloud = np.reshape(trk_cloud, (num_streamlines*num_points, 3))

    #####################
    # Data augmentation #
    #####################
    if is_test == False:
        # Rotation factors
        x_angle = np.random.uniform(-np.pi/4, np.pi/4)
        y_angle = np.random.uniform(-np.pi/4, np.pi/4)
        z_angle = np.random.uniform(-np.pi/4, np.pi/4)

        # Scale factors
        x_factor = np.random.uniform(0.9, 1.5)
        y_factor = np.random.uniform(0.9, 1.5)
        z_factor = np.random.uniform(0.9, 1.5)

        # Displacement factors
        x_disp = np.random.uniform(-0.1,0.1)
        y_disp = np.random.uniform(-0.1,0.1)
        z_disp = np.random.uniform(-0.1,0.1)

        # Noise stdev factor
        noise_stdev = np.random.uniform(0,0.02)

        # Get the matrices
        rot_matrix = get_rot_matrix(x_angle, y_angle, z_angle)
        scale_matrix = get_scale_matrix(x_factor, y_factor, z_factor)

        # Augment the TOM cloud
        tom_cloud = rotate_tom_cloud(tom_cloud, rot_matrix)
        tom_cloud = displace_tom_cloud(tom_cloud, x_disp, y_disp, z_disp)
        tom_cloud = scale_tom_cloud(tom_cloud, scale_matrix)
        tom_cloud = tom_add_noise(tom_cloud, 0, noise_stdev)

        # Augment the TRK cloud
        trk_cloud = rotate_trk_cloud(trk_cloud, rot_matrix)
        trk_cloud = displace_trk_cloud(trk_cloud, x_disp, y_disp, z_disp)
        trk_cloud = scale_trk_cloud(trk_cloud, scale_matrix)

    # Extract seeds from resulting tractogram
    seeds = np.reshape(trk_cloud, (num_streamlines, num_points, 3))[:,0,:]

    # Convert to torch tensors
    tom = torch.from_numpy(np.float32(tom_cloud))
    tom = tom.permute(1,0) # channels first for pytorch

    tractogram = torch.from_numpy(np.float32(trk_cloud))
    tractogram = tractogram.permute(1, 0) # channels first for pytorch

    seeds = torch.from_numpy(np.float32(seeds))
    seeds = seeds.permute(1,0)

    return [[tom, seeds], tractogram]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, toms_dir, tractograms_dir, is_test=False):
        # Get lists of files
        self.toms_fn = glob(toms_dir + '/*.npy')
        self.tractograms_fn = glob(tractograms_dir + '/*.npy')

        # Sort for correct matching between the sets of filenames
        self.toms_fn.sort()
        self.tractograms_fn.sort()

        self.is_test = is_test
        
        # Load data into RAM
        #self.data = []
        #print("Loading dataset into RAM...")
        #for i in range(len(self.toms_fn)):
        #    self.data.append(get_data(self.toms_fn[i], self.tractograms_fn[i]))
        #    print(i)

    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):
        return get_data(self.toms_fn[idx], self.tractograms_fn[idx], self.is_test)

    def __len__(self):
        return len(self.toms_fn)

def OutputToPoints(output):
    points = output.permute(1,0)
    points = points.cpu().detach().numpy()

    return points

def OutputToStreamlines(output):
    streamlines = output
    streamlines = streamlines.permute(1, 0) # from (3,N) to (N,3)
    streamlines = streamlines.cpu().detach().numpy()

    streamlines = np.reshape(streamlines, (-1,num_points,3))

    return streamlines
    
