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
from models.sampling import CustomDataset, OutputToStreamlines, OutputToPoints
args = sys.argv
model_name = args[1]
epoch_number = args[2]
num_sl = 128
points_per_sl = 10

#test_dir = '../data/CST_left_128_10_partial/not_preprocessed/test'
#test_dir = '../data/CST_left_original_10_ppsl'
#test_dir = '../data/multitract_10_ppsl'
#test_dir = '../data/multitract_10_ppsl/test'
test_dir = '../data/multitract_10_ppsl/unseen'

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
        if type(inputs) is list:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)

        if type(labels) is list:
            for i in range(len(labels)):
                labels[i] = labels[i].to(device)
        else:
            labels = labels.to(device)

        for i in range(1):
            #inputs[1] += ((torch.rand(1,3,128) - 0.5)*2*0.01).to(device)
            if type(inputs) is list:
                outputs = model.forward(*inputs)
            else:
                outputs = model.forward(inputs)


            if i == 0:
                label, output = labels[0], outputs[0]
            else:
                output = torch.cat((output, outputs[0]), dim=-1)
                label = labels[0]

        #output = torch.unique(output, dim=0)
        #label = torch.unique(label, dim=0)
        print(output.size(), label.size())
        #matplotlib_tom_cloud(inputs[0][0])
        plotly_tom(inputs[0][0])
        #plotly_cloud(output, OutputToPoints)
        #plotly_cloud(label, OutputToPoints)
        plotly_lines(output, OutputToStreamlines)
        plotly_lines(label, OutputToStreamlines)
        #plotly_tom_and_trk(inputs[0], output, OutputToPoints)
        #plotly_everything(inputs[0][0], output, OutputToPoints, num_sl, points_per_sl)
        #plotly_everything(inputs[0][0], label, OutputToPoints, num_sl, points_per_sl)
        #plot_output_maps(output, 'output', OutputToPoints, num_sl, points_per_sl)
        #plot_output_maps(label, 'label', OutputToPoints, num_sl, points_per_sl)
        #matplotlib_lines(output, OutputToStreamlines)
        #matplotlib_combined_lines(output, label, OutputToStreamlines)
        #matplotlib_overlap_lines(output, label, OutputToStreamlines)
        input('continue')

        subject_count += 1
