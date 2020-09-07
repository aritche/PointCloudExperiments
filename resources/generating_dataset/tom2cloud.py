import numpy as np
import cv2
import nibabel as nib
from dipy.io.image import load_nifti 
from glob import glob
import os

def point2cloud(tom_fn, out_prefix):
    tom = nib.load(tom_fn).get_data()

    summed = np.sum(np.abs(tom),axis=-1)
    coords = np.nonzero(summed) # shape: (n, n, n)
    vectors = tom[coords] # shape: (n,3)
    coords = np.stack([coords[0]/145, coords[1]/174, coords[2]/145],axis=-1)
    combined = np.concatenate((coords,vectors), axis=-1)
    np.save('./vector_clouds/' + out_prefix + '.npy', combined)
    print("\t>" + str(combined.shape))


for fn in glob('../../data/CST_left_128_10_partial/not_preprocessed/TOMs/*.nii.gz'):
    prefix = os.path.split(fn)[-1].split('.')[0]
    print(prefix)
    point2cloud(fn, prefix)
