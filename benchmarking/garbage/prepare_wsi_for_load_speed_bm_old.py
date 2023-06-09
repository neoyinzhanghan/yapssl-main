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
import subprocess
from openslide import open_slide

from yatt.MaskAndPatch import MaskAndPatch
from yatt.read_image import get_patch_at_mpp

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

args = parser.parse_args()


#########################################################
# NON-COMMAND-LINE PARAMETERS
#########################################################

patch_ext = '.png'


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
# SAVE THE LIST PICKLE FILE CONTAINING THE SVS FILENAMES
#########################################################

svs_lst = random_svs_files(args.wsi_dir, num_files=args.n_wsis)

def _get_roots(svs_lst: list) -> list:
    """ Return a list containing only the root of each filenames in sys_lst. """

    root_lst = []
    for name in svs_lst:
        name = os.path.basename(name)
        root, extension = os.path.splitext(name)

        root_lst.append(root)

    return root_lst

root_lst = _get_roots(svs_lst)

lst_path = os.path.join(output_dir, 'wsi_fname_roots.pkl')
with open(lst_path, 'wb') as file:
    pickle.dump(root_lst, file)


#########################################################
# USEFUL FUNCTIONS FOR PARQUET CONVERSION
#########################################################

def _flatten_patcher(patcher, wsi, mpp=args.mpp):
    """ Return a panda data frame that stores all the patches in flattened format. """
    """ grad_val | x | y | channel | patch_idx 
        _________|___|___|_________|____________
                 |   |   |         |         
                 |   |   |         |            
    """
    patch_idxs = np.arange(patcher.patch_coords_level0_.shape[0])
    patch_idx = patch_idxs[patcher.patch_tissue_prop_ >= 0]

    total = len(patch_idx)

    df_dct = {}

    for i in range(total):
        idx = patch_idx[i]
        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=mpp,
                                 coords_level0=patcher.patch_coords_level0_[idx],
                                 patch_size_level0=patcher.patch_size_level0_)

        patch_np = np.array(patch)
        patch_flat = patch_np.flatten()
        # for channel in range(n_channels): # TODO reimplement this using the numpy default flattening operation instead of doing it manually
        #     for x in range(patch_size_x):
        #         for y in range(patch_size_y):
        #             grad_val = patch_np[x][y][channel]
        #             df_dct['grad_val'].append(grad_val)
        #             df_dct['x'].append(x)
        #             df_dct['y'].append(y)
        #             df_dct['channel'].append(channel)
        #             df_dct['idx'].append(idx)

        df_dct[str(idx)] = patch_flat

    #     print('Parquet Packing Progress -- ' + str(int(100*(i + 1)/total)) + '%') # TODO to remove after debugging
    #
    # print('Converting to pandas dataframe...')
    df = pd.DataFrame(df_dct)

    return df


def _tensorize_patch_df(df):
    """ Return the pytorch tensor that the input dataframe is meant to represent. """
    # TODO this may not be very important for benchmarking purposes of loading speed
    pass

#########################################################
# THIS IS JUST FOR FUN, ALWAYS WANTED TO DO THIS
# THE ULTIMATE EXAMPLE OF SINFUL PREMATURE OPTIMIZATION
#########################################################

def spinning_wheel_H5(stop_event):
    spinner = ['\\________/','/\\________', '__/\\______','____/\\____', '______/\\__', '________/\\']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"Preparing H5 -- {spinner[i % len(spinner)]}\r")
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        i += 1

def spinning_wheel_for_loop(stop_event, percentage, name):
    spinner = ['\\________/','/\\________', '__/\\______','____/\\____', '______/\\__', '________/\\'] # ['==', '\\\\', '||', '//']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"WSI {name} Completed -- {percentage}% -- {spinner[i % len(spinner)]}\r")
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        i += 1


#########################################################
# TRAVERSE THROUGH EACH SVS IMAGE TO CREATE EACH DATATYPE
# AND SAVE THEM IN THE RESPECTIVE FOLDERS
#########################################################


print('Preparing SVS, FOLDER, PARQUET Patches')
for ind in range(len(svs_lst)):

    percentage = int((ind + 1) / len(svs_lst) * 100)
    svs_fname = svs_lst[ind]




    # SVS -- easy, just copy and pasting

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinning_wheel_for_loop, args=(stop_event, percentage, 'SVS'))
    spinner_thread.start()

    base_svs_fname = os.path.basename(svs_fname)
    root, extension = os.path.splitext(base_svs_fname)
    destination_svs_fpath = os.path.join(svs_dir, base_svs_fname)

    shutil.copy2(svs_fname, destination_svs_fpath)

    stop_event.set()
    spinner_thread.join()



    # FOLDER

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinning_wheel_for_loop, args=(stop_event, percentage, 'FOLDER'))
    spinner_thread.start()

    patches_folder = os.path.join(folder_dir, root)
    os.makedirs(patches_folder, exist_ok=True)

    wsi = open_slide(svs_fname)
    patcher = MaskAndPatch(mpp4mask=10)
    patcher.fit_mask_and_patch_grid(wsi)

    patch_idxs = np.arange(patcher.patch_coords_level0_.shape[0])
    patch_idx = patch_idxs[patcher.patch_tissue_prop_ >= 0]
    # TODO Want ALL of the patches, and also don't worry too much about positional embedding at this moment

    for idx in patch_idx:
        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=args.mpp,
                                 coords_level0=patcher.patch_coords_level0_[idx],
                                 patch_size_level0=patcher.patch_size_level0_)
        file_name = root + '_patch_' + str(idx) + patch_ext
        file_path = os.path.join(patches_folder, file_name)
        patch.save(file_path)

    stop_event.set()
    spinner_thread.join()



    # PARQUET -- first flatten a patch-batch tensor into a pande dataframe and have a method of converting it back

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinning_wheel_for_loop, args=(stop_event, percentage, 'PARQUET'))
    spinner_thread.start()

    # TODO make sure to save a meta-data file, which also records the position of the patches -- not important rn

    parquet_path = os.path.join(parquet_dir, root + '.parquet')
    flat_patch_df = _flatten_patcher(patcher, wsi)

    flat_patch_df.to_parquet(parquet_path, engine='auto', compression='snappy')

    stop_event.set()
    spinner_thread.join()



    # TODO need to flatten a big patch tensor which is 224 x 224 x 3 x n_patches shaped tensor
    # TODO can be flatten into a 2 flat dataframe in Panda and need to write a reverse flatting code
    # TODO for loading purposes it might be okay to save the flattened version and move on

    # TODO META DATA <<< can skip for now to avoid premature optimization
    # TODO make sure to save metadata together
    # TODO save the number of patches in the svs
    # TODO we are eventually interested in loading with parallelization <<< figure out how to do

# H5 -- need to run the SVS file through yatt -- this is already implemented as a yatt script
subprocess.call([
    'python', args.create_wsi_patch_h5_script_path,
    '--data_dir', svs_dir,
    '--save_dir', h5_dir,
    '--mpp', str(args.mpp)
]) # TODO am i doing this the right way? feeling like it is a little bit criminal to do it like this

# TODO need to save the number of patches in the pickle file... preferably