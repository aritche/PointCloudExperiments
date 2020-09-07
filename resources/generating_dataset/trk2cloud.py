import numpy as np
from nibabel import trackvis
from dipy.io.image import load_nifti 
from glob import glob
import os
from dipy.tracking import utils

def trk2cloud(trk_fn, out_prefix, inverse_affine):
    # Open the file
    streamlines, header = trackvis.read(trk_fn)
    streamlines = [s[0] for s in streamlines]
    streamlines = np.array(streamlines)

    # Convert to voxel space and normalise
    streamlines = utils.apply_affine(aff=inverse_affine, pts=streamlines)
    coords = np.reshape(streamlines, (-1,3))
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    x, y, z = x/145, y/174, z/145
    coords = np.stack([x,y,z], axis=-1)

    np.save('./tractogram_clouds/' + out_prefix + '.npy', coords)

_, affine = load_nifti('../../data/CST_left_128_10_partial/not_preprocessed/TOMs/599469_0_CST_left_DIRECTIONS.nii.gz')
inverse_affine = np.linalg.inv(affine)

for fn in glob('../../data/CST_left_128_10_partial/not_preprocessed/TOMs/*.nii.gz'):
    prefix = os.path.split(fn)[-1].split('.')[0].split('_')[:-1]
    prefix = '_'.join(prefix)
    print(prefix)
    fn = '../../data/CST_left_128_10_partial/not_preprocessed/tractograms/' + prefix + '.trk'
    trk2cloud(fn, prefix, inverse_affine)
