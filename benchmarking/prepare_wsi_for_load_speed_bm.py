#########################################################
# RELEVANT PACKAGE IMPORTING
#########################################################
import sys
import time
import threading
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil
from openslide import open_slide
from tqdm import tqdm

from yatt.MaskAndPatch import MaskAndPatch
from yatt.tissue_mask import RgbTissueMasker
from yatt.read_image import get_patch_at_mpp
from yatt.h5_utils import add_array_batch_to_h5

#########################################################
# ARGUMENT PARSING
#########################################################

parser = argparse.ArgumentParser()

#########################################################
group = parser.add_argument_group('Bureaucracy')
#########################################################

group.add_argument('--wsi_dir', type=str,
                   help='directory storing a bank of .svs WSIs')
group.add_argument('--n_wsis', type=int,
                   help='number of WSIs to be proceeded in the benchmarking task')
group.add_argument('--bm_dir_location', type=str,
                   help='the location to store the output bm_dir')
group.add_argument('--create_wsi_patch_h5_script_path', type=str,
                   help='the location of the script which extract h5 patches')


#########################################################
group = parser.add_argument_group('WSI info')
#########################################################

group.add_argument('--mpp', type=float, default=0.57142857142,
                   help='the mpp at which patching is performed')

group.add_argument('--patch_size', type=int, default=224,
                   help='the size of the patches')

group.add_argument('--min_tissue_area', type=int, default=1000,
                   help='the minimum tissue area of selected patches')

group.add_argument('--patch_ext', type=str, default='.png',
                   help='the file format of the saved patch images into folders')

args = parser.parse_args()


#########################################################
# FIND THE SVS FILES AND RANDOMLY SUBSET
#########################################################

def random_svs_files(directory, num_files=5):
    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out the .svs files
    svs_files = [file for file in all_files if file.endswith('.svs')]

    # Take a random subset of the .svs files
    random_subset = random.sample(svs_files, min(num_files, len(svs_files)))

    # Create a list of full file paths
    file_paths = [os.path.join(directory, file) for file in random_subset]

    return file_paths


#########################################################
# CREATE AN EMPTY FOLDER FOR THE WSIs
#########################################################

output_dir = os.path.join(args.bm_dir_location, 'bm_dir')
os.makedirs(output_dir, exist_ok=True)


#########################################################
# CREATE AN EMPTY FOLDER FOR EACH DATATYPE
#########################################################

svs_dir = os.path.join(output_dir, 'SVS')
h5_dir = os.path.join(output_dir, 'H5')
folder_dir = os.path.join(output_dir, 'FOLDER')
parquet_dir = os.path.join(output_dir, 'PARQUET')

os.makedirs(svs_dir, exist_ok=True)
os.makedirs(h5_dir, exist_ok=True)
os.makedirs(folder_dir, exist_ok=True)
os.makedirs(parquet_dir, exist_ok=True)


#########################################################
# SAVE THE LIST PICKLE FILE CONTAINING THE FILENAMES ROOTS
#########################################################

svs_lst = random_svs_files(args.wsi_dir, num_files=args.n_wsis)

def _get_roots(svs_lst: list) -> list:
    """ Return a list containing only the root of each filename in sys_lst. """

    root_lst = []
    for name in svs_lst:
        name = os.path.basename(name)
        root, extension = os.path.splitext(name)

        root_lst.append(root)

    return root_lst

root_lst = _get_roots(svs_lst)

meta_path = os.path.join(output_dir, 'bm_meta.pkl')

meta = {'wsi_root_lst': root_lst,
        'mpp': args.mpp,
        'patch_size': args.patch_size,
        'min_tissue_area': args.min_tissue_area,
        'patch_ext': args.patch_ext}

with open(meta_path, 'wb') as file:
    pickle.dump(meta, file)


#########################################################
# USEFUL FUNCTIONS FOR PARQUET CONVERSION
#########################################################

def save_patches_parquet(wsi,
                         patch_coords,
                         mpp: float,
                         patch_size: int,
                         fpath: str,
                         tag: str = ''
                         ):
    """ Return a panda data frame that stores all the patches in flattened format. """

    df_dct = {}

    for (idx, coords_level0) in enumerate(tqdm(patch_coords, desc='patches parquet -- '+tag, position=1)):
        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=mpp,
                                 coords_level0=coords_level0,
                                 patch_size_mpp=patch_size,
                                 # tol=self.tol
                                 )

        patch_np = np.array(patch)
        patch_flat = patch_np.flatten()
        df_dct[str(idx)] = patch_flat

    df = pd.DataFrame(df_dct)

    df.to_parquet(fpath, engine='auto', compression='snappy')


#########################################################
# USEFUL FUNCTIONS FOR FOLDER SAVING
#########################################################

def save_patches_folder(wsi,
                        patch_coords,
                        mpp: float,
                        patch_size: int,
                        folder: str,
                        ext: str = '.png',
                        tag: str = ''):

    """
    Save all the patches of the wsi into folder with extention ext

    Parameters
    ----------
    wsi: the whole slide image

    patch_coords: array-like, shape (n_patches, 2)
        The patch coordinates at level 0.

    mpp:
        mpp to load the patches in at.

    patch_size:
        Size of the patches:

    folder:
        File path to folder where images will be saved.

    ext:
        What type of image save.
    """

    os.makedirs(folder, exist_ok=True)

    for (idx, coords_level0) in enumerate(tqdm(patch_coords, desc='folder of patches -- '+tag, position=1)):
        # Load patch from svs file
        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=mpp,
                                 coords_level0=coords_level0,
                                 patch_size_mpp=patch_size,
                                 # tol=self.tol
                                 )

        # save patch
        patch_fpath = os.path.join(folder, 'patch_{}{}'.format(idx, ext))
        patch.save(patch_fpath)


#########################################################
# USEFUL FUNCTIONS FOR H5 SAVING
#########################################################

def save_patches_h5(wsi,
                    patch_coords,
                    mpp: float,
                    patch_size: int,
                    fpath: str,
                    delete_if_exists: bool = True,
                    tag: str = ''):
    """
    Save all the patches of the wsi as an H5 file

    Parameters
    ----------
    wsi:

    patch_coords: array-like, shape (n_patches, 2)
        The patch coordinates at level 0.

    mpp:
        mpp to load the patches in at.

    patch_size:
        Size of the patches:

    fpath:
        File path where to save the h5 file.

    delete_if_exists: bool
        Delete file if one already exits
    """

    # delete file if it already exists
    if delete_if_exists and os.path.exists(fpath):
        os.remove(fpath)

    for coords_level0 in tqdm(patch_coords, desc='patches h5 -- ' + tag, position=1):
        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=mpp,
                                 coords_level0=coords_level0,
                                 patch_size_mpp=patch_size,
                                 # tol=self.tol
                                 )

        add_array_batch_to_h5(fpath=fpath,
                              data=np.array(patch),
                              name='imgs',
                              is_single_obs=True)


#########################################################
# TRAVERSE THROUGH EACH SVS IMAGE TO CREATE EACH DATATYPE
# AND SAVE THEM IN THE RESPECTIVE FOLDERS
#########################################################

print('Preparing SVS, FOLDER, PARQUET Patches')
for ind in range(len(svs_lst)):

    tag = str(ind+1) + '/' + str(args.n_wsis)
    svs_fname = svs_lst[ind]


    # SVS -- easy, just copy and pasting

    base_svs_fname = os.path.basename(svs_fname)
    root, extension = os.path.splitext(base_svs_fname)
    destination_svs_fpath = os.path.join(svs_dir, base_svs_fname)

    shutil.copy2(svs_fname, destination_svs_fpath)

    # Prepare the patch grid

    wsi = open_slide(svs_fname)

    patcher = MaskAndPatch(mpp4mask=10,
                           tissue_masker=RgbTissueMasker(),
                           mpp4patch=args.mpp,
                           patch_size=args.patch_size,
                           store_mask=True
                           )

    patcher.fit_mask_and_patch_grid(wsi)
    patch_df = patcher.get_patch_df()


    # Only include patches with enough tissue

    idxs_tissue = patch_df.query("tissue_area >= {}".
                                 format(args.min_tissue_area)).index.values
    patch_df['include'] = False
    patch_df.loc[idxs_tissue, 'include'] = True

    n_tissue_patches = len(idxs_tissue)
    n_total_patches = patch_df.shape[0]

    # subset to only the patches we will include
    patch_df_included = patch_df.query('include')
    patch_coords = patch_df_included[['x', 'y']].values


    # FOLDER

    patches_folder = os.path.join(folder_dir, root)
    os.makedirs(patches_folder, exist_ok=True)

    save_patches_folder(wsi=wsi,
                        patch_coords=patch_coords,
                        mpp=args.mpp,
                        patch_size=args.patch_size,
                        folder=patches_folder,
                        tag=tag)

    # TODO Want ALL of the patches, and also don't worry too much about positional embedding at this moment

    # PARQUET -- first flatten a patch-batch tensor into a panda dataframe and have a method of converting it back

    # TODO make sure to save a meta-data file, which also records the position of the patches -- not important rn

    parquet_path = os.path.join(parquet_dir, root + '.parquet')

    save_patches_parquet(wsi=wsi,
                         patch_coords=patch_coords,
                         mpp=args.mpp,
                         patch_size=args.patch_size,
                         fpath=parquet_path,
                         tag=tag)

    # H5

    h5_fpath = os.path.join(h5_dir, root + '.h5')
    save_patches_h5(wsi=wsi,
                    patch_coords=patch_coords,
                    mpp=args.mpp,
                    patch_size=args.patch_size,
                    fpath=h5_fpath,
                    tag=tag)