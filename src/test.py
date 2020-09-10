"""
Script for testing a model
"""

#from resources.vis import VisdomLinePlotter
import visdom

from dipy.tracking import utils
from dipy.io.image import load_nifti 
import torch
import numpy as np
from dipy.io.streamline import load_trk, save_trk
from torchsummary import summary

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import cv2

from plotly import graph_objs as go
import plotly.express as px

import sys
import random

from nibabel import trackvis

from analysis import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dump = input("Please make sure you are importing the correct model into the test file...")

# Parameters
from models.invariant_conv import CustomDataset, OutputToStreamlines, OutputToPoints
args = sys.argv
model_name = args[1]
epoch_number = args[2]
num_sl = 128
points_per_sl = 10

test_dir = '../data/CST_left_128_10_partial/not_preprocessed/test'

# Load model
print("Loading model...")
fn = './results/' + model_name + '/epoch_' + epoch_number + '.pth'
model = torch.load(fn)

#summary(model, (3, 144, 144, 144))

print('Sending model to device...')
model.to(device)
model.eval()

# Load dataset
print('Loading dataset...')
toms_dir = test_dir + '/vector_clouds'
tractograms_dir = test_dir + '/tractogram_clouds'
#_, affine = load_nifti(toms_dir + '/644044_0_CST_left.nii.gz')
#inverse_affine = np.linalg.inv(affine)
dataset = CustomDataset(toms_dir, tractograms_dir, is_test=True)
testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

torch.cuda.empty_cache()

subject_count = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)

        label, output = labels[0], outputs[0]

        if subject_count == 0 or subject_count == 144:
            print(outputs.size())
            #plotly_cloud(output, OutputToPoints)
            #plotly_cloud(label, OutputToPoints)
            plot_output_maps(output, OutputToPoints, num_sl, points_per_sl)
            plot_output_maps(label, OutputToPoints, num_sl, points_per_sl)
        else:
            print(subject_count)

        subject_count += 1
