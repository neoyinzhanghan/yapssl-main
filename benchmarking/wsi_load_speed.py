#########################################################
# RELEVANT PACKAGE IMPORTING
#########################################################
import argparse
import time
import numpy as np
import pickle
import os
from openslide import open_slide
import pandas as pd
import h5py
import glob
from tqdm import tqdm
import torch
from yatt.read_image import get_patch_at_mpp
from yatt.MaskAndPatch import MaskAndPatch
from yatt.tissue_mask import RgbTissueMasker
from PIL import Image

#########################################################
# ARGUMENT PARSING
#########################################################

parser = argparse.ArgumentParser()

#########################################################
group = parser.add_argument_group('Directory Setting')
#########################################################

group.add_argument('--bm_dir', type=str,
                   help='folder containing the same wsi saved in different data format, should be the direct output of prepare_wsi_for_load_speed_bm.py')

group.add_argument('--patch_ext', type=str, default='.png',
                   help='the file format of the saved patch images into folders')

args = parser.parse_args()


#########################################################
# EXTRACT META-DATA
#########################################################

meta_path = os.path.join(args.bm_dir, 'bm_meta.pkl')
with open(meta_path, 'rb') as file:
    meta = pickle.load(file)

wsi_fnames = meta['wsi_root_lst']
mpp = meta['mpp']
patch_size = meta['patch_size']
min_tissue_area = meta['min_tissue_area']
patch_ext = meta['patch_ext']

#########################################################
# THE UPLOAD FUNCTION FOR ALL RELEVANTE FILE TYPES
#########################################################

def _get_patch_coords(fpath: str,
                      mpp: float,
                      patch_size: int,
                      min_tissue_area: int):

    wsi = open_slide(fpath)
    patcher = MaskAndPatch(mpp4mask=10,
                           tissue_masker=RgbTissueMasker(),
                           mpp4patch=mpp,
                           patch_size=patch_size,
                           store_mask=True
                           )

    patcher.fit_mask_and_patch_grid(wsi)
    patch_df = patcher.get_patch_df()


    # Only include patches with enough tissue

    idxs_tissue = patch_df.query("tissue_area >= {}".
                                 format(min_tissue_area)).index.values
    patch_df['include'] = False
    patch_df.loc[idxs_tissue, 'include'] = True

    # subset to only the patches we will include
    patch_df_included = patch_df.query('include')
    patch_coords = patch_df_included[['x', 'y']].values

    return patch_coords

def load_patches_openslide(fpath: str,
                           patch_coords: list,
                           mpp: float,
                           patch_size: int):
    """
    Parameters
    ----------
    fpath:
        Path to the WSI.

    Output
    -------
    patches: array-like, shape (n_patches, patch_height, patch_width, n_channels)
    """

    wsi = open_slide(fpath)

    patches = []
    for coords_level0 in patch_coords:

        patch = np.array(get_patch_at_mpp(wsi=wsi,
                                          mpp=mpp,
                                          coords_level0=coords_level0,
                                          patch_size_mpp=patch_size,
                                          # tol=self.tol
                                          ))
        patch = torch.from_numpy(patch)
        patches.append(patch)

    return torch.stack(patches)

def load_patches_h5(fpath: str):
    """
    Parameters
    ----------
    fpath:
        Path to the h5 file.

    Output
    -------
    patches: array-like, shape (n_patches, patch_height, patch_width, n_channels)
    """
    with h5py.File(fpath, 'r') as f:
        return torch.from_numpy(np.array(f['imgs']))

def load_patches_folder(folder: str,
                        patch_ext:str = '.png'):
    """
    Parameters
    ----------
    folder:
        Path to folder where images are saved.

    Output
    ------
    patches:
        The patches; one of numpy array, list of PIL, or torch tensor

    """
    # Load each image
    patches = []

    img_files = glob.glob(os.path.join(folder, '*'+patch_ext))
    for img_file in img_files:
        patch = Image.open(img_file)
        patch = np.array(patch)
        patch = torch.from_numpy(patch) # Read the image as a pytorch tensor

        patches.append(patch)

    return torch.stack(patches)


def load_patches_parquet(fpath:str, patch_size:int):
    df = pd.read_parquet(fpath)

    patches = []
    for idx in df.columns:
        col = df[idx].values
        patch = col.reshape(patch_size, patch_size, 3)
        patch = torch.from_numpy(patch)
        patches.append(patch)

    return torch.stack(patches)


def _load_wsi(fpath:str) -> float:
    """ Based on the extension of the file, it uses different methods to load the file and return the loading time. """

    root, extension = os.path.splitext(fpath)

    if extension == '.svs':
        patch_coords = _get_patch_coords(fpath=fpath,
                                         mpp=mpp,
                                         patch_size=patch_size,
                                         min_tissue_area=min_tissue_area)

        start_time = time.time()
        tens = load_patches_openslide(fpath=fpath,
                                      patch_coords=patch_coords,
                                      mpp=mpp,
                                      patch_size=patch_size)

        end_time = time.time()

    elif extension == '.h5':
        # This can just be done using the h5py library

        start_time = time.time()
        # Open the HDF5 file in read mode
        tens = load_patches_h5(fpath=fpath)
        end_time = time.time()

    elif extension == '.parquet':
        # I think the parquet library has a natual notion of loading
        # But need to load and then unflatten into the correct tensor shape

        start_time = time.time()
        tens = load_patches_parquet(fpath,
                                    patch_size=patch_size)

        end_time = time.time()

    else:
        # Does the mean loading each patch one by one through traversal?

        start_time = time.time()
        tens = load_patches_folder(fpath, patch_ext=patch_ext)

        end_time = time.time()

    loading_time = end_time - start_time

    return loading_time

#########################################################
# BENCHMARKING SPEED
#########################################################

svs_dir = os.path.join(args.bm_dir, 'SVS')
h5_dir = os.path.join(args.bm_dir, 'H5')
folder_dir = os.path.join(args.bm_dir, 'FOLDER')
parquet_dir = os.path.join(args.bm_dir, 'PARQUET')

header = ['root', 'SVS', 'H5', 'FOLDER', 'PARQUET', 'n_patches']
time_matrix = []

for root in tqdm(wsi_fnames):
    svs_path = os.path.join(svs_dir, root + '.svs')
    h5_path = os.path.join(h5_dir, root + '.h5')
    folder_path = os.path.join(folder_dir, root + '')
    parquet_path = os.path.join(parquet_dir, root + '.parquet')

    svs_time = _load_wsi(svs_path)
    h5_time = _load_wsi(h5_path)
    folder_time = _load_wsi(folder_path)
    parquet_time = _load_wsi(parquet_path)

    n_patches = len(glob.glob(os.path.join(folder_path, '*.png')))

    time_row = [root, str(svs_time), str(h5_time), str(folder_time), str(parquet_time), str(n_patches)]

    time_matrix.append(time_row)

matrix = np.array(time_matrix)
header = ','.join(header)
output_file = 'wsi-load-speed.csv'
output_path = os.path.join(args.bm_dir, output_file)

np.savetxt(fname=output_path,
           X=matrix,
           delimiter=',',
           fmt='%s',
           header=header,
           comments='')