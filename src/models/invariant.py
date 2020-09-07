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

from chamferdist import ChamferDistance

from dipy.io.streamline import load_trk
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from nibabel import trackvis
from dipy.tracking import utils

num_points = 10
num_streamlines = 128
w_coords = 1 # weight for the coords loss
w_seeds =  1 # weight for the seeds loss

# Model adapted from https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.25)
                                                                                                                 # VOLUME SIZE                      # PARAMETERS
        # Encoding (input -> 512 vector)                                                                         # 3 x 144 x 144 x 144 -> 8.9M      (IN * F^3 + 1)*OUT
        self.mlp_1 = nn.Conv1d(in_channels=6,  out_channels=64, kernel_size=1)
        self.batchnorm_1 = nn.BatchNorm1d(64)
        self.mlp_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.batchnorm_2 = nn.BatchNorm1d(128)
        self.mlp_3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.batchnorm_3 = nn.BatchNorm1d(1024)

        #self.maxpool = nn.MaxPool1d(1024)

        # 128 x 10 x 3
        self.linear_1 = nn.Linear(in_features=1024, out_features=1280)
        self.batchnorm_4 = nn.BatchNorm1d(1280)
        self.linear_2 = nn.Linear(in_features=1280, out_features=3840)
        #self.batchnorm_5 = nn.BatchNorm1d(3840)

    #def forward(self, x, seeds, fixed_encoding):
    def forward(self, x):
        # Input
        #local_features = self.dropout(self.relu(self.batchnorm_1(self.mlp_1(x))))
        #x = self.dropout(self.relu(self.batchnorm_2(self.mlp_2(local_features))))
        #x = self.dropout(self.relu(self.batchnorm_3(self.mlp_3(x))))

        local_features = self.dropout(self.relu(self.mlp_1(x)))
        x = self.dropout(self.relu(self.mlp_2(local_features)))
        x = self.dropout(self.relu(self.mlp_3(x)))

        # Max pooling
        #x = self.maxpool(x)
        x = nn.MaxPool1d(x.size(-1))(x)
        global_features = x.view(-1, 1024)

        # Concat global and local features
        # 64x1000 vs. 1024
        #global_features = global_features.unsqueeze(2)
        #global_features = global_features.expand(-1, 1024, local_features.size(-1))
        #x = torch.cat((local_features, global_features), dim=1)

        # Generate the final result
        #print(x.size())
        # 2x1088x1000
        #x = self.dropout(self.relu(self.batchnorm_4(self.linear_1(global_features))))
        x = self.dropout(self.relu(self.linear_1(global_features)))
        #print(x.size())
        #x = self.sigmoid(self.linear_2(x))
        x = self.linear_2(x)
        #print(x.size())

        #result = x.view(-1, 3, num_streamlines*num_points)
        result = x.view(-1, 3*num_points, num_streamlines)

        #x = x.view(-1, 512)     # Output: (2048)
        #x = fixed_encoding
        #encoding = x.clone()
        #return [p,encoding]

        return result

# Custom loss function
def CustomLoss(output, target):
    output = output.permute(0,2,1)
    target = target.permute(0,2,1)
    #output = output.view(-1,num_streamlines, 3)
    #target = target.view(-1,num_streamlines, 3)

    # Re-implemented MSE loss for efficiency reasons
    chamferDist = ChamferDistance()
    dist, _, _, _ = chamferDist(output, target)

    return dist.mean()
    #return ((output - target)**2).mean()

def get_data(tom_fn, tractogram_fn):
    # Load data
    tom = np.load(tom_fn)
    tractogram = np.float32(np.load(tractogram_fn))

    #####################
    # Data augmentation #
    #####################
    # 1. Add noise to TOM orientations
    noise_stdev = np.random.rand(1) * 0.05
    #noise = np.random.normal()
    #noise_stdev = torch.rand(1) * 0.05
    #noise = torch.normal(mean=torch.zeros(tom.size()), std=torch.ones(tom.size())*noise_stdev)
    #noise[] # we want the first 3 dimensions to be 1
    #tom += noise

    # 2. ROTATION

    # 3. DISPLACEMENT

    # 4. ELASTIC DEFORMATION

    # 5. Zooming

    # 6. Resampling input pointcloud

    # Convert to torch tensors
    tom = torch.from_numpy(np.float32(tom))
    tom = tom.permute(1,0) # channels first for pytorch

    tractogram = np.reshape(tractogram, (-1,num_points*3))
    tractogram = torch.from_numpy(tractogram)
    tractogram = tractogram.permute(1, 0) # channels first for pytorch

    return [tom, tractogram]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, toms_dir, tractograms_dir):
        # Get lists of files
        self.toms_fn = glob(toms_dir + '/*.npy')
        self.tractograms_fn = glob(tractograms_dir + '/*.npy')

        # Sort for correct matching between the sets of filenames
        self.toms_fn.sort()
        self.tractograms_fn.sort()
        
        # Load data into RAM
        #self.data = []
        #print("Loading dataset into RAM...")
        #for i in range(len(self.toms_fn)):
        #    self.data.append(get_data(self.toms_fn[i], self.tractograms_fn[i]))
        #    print(i)

    # Given an index, return the loaded [data, label]
    def __getitem__(self, idx):
        return get_data(self.toms_fn[idx], self.tractograms_fn[idx])

    def __len__(self):
        return len(self.toms_fn)

def OutputToStreamlines(output):
    streamlines = output
    streamlines = streamlines.permute(1, 0) # from (3,N) to (N,3)
    streamlines = streamlines.cpu().detach().numpy()

    streamlines = np.reshape(streamlines, (num_streamlines,num_points,3))

    return streamlines
    
