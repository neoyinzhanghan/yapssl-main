import pandas as pd
import os
import argparse
import glob
import random

from benp.make_patch_grid import make_patch_grid

from benp.save_load_patches import load_patches_openslide,\
    save_patches_h5, load_patches_h5,\
    save_patches_folder, load_patches_folder

from benp.save_load_patches_par import load_patches_openslide_par,\
    load_patches_h5_par, \
    save_patches_folder_par, load_patches_folder_par

from benp.save_load_ray import load_patches_folder_ray

from benp.save_load_patches import iter_patches_openslide

from benp.utils import get_file_size, get_folder_size

from benp.folder_saver import PilImageSaver, NumpySaver, Bag2CsvSaver, Bag2ParquetSaver, TorchSaver,\
    save_in_folder, save_in_folder_par,\
    load_from_folder, load_from_folder_par, safe_delete_folder
from yatt.RuntimeTracker import RuntimeTracker

###########################################################
# ARGUMENT PARSING
###########################################################

parser = argparse.ArgumentParser()

###########################################################
group = parser.add_argument_group('Bureaucracy')
###########################################################

group.add_argument('--save_dir', type=str,
                   help='where do you want to save your WSI data')
group.add_argument('--results_dir', type=str,
                   help='where do you want to save your results')
group.add_argument('--wsi_dir', type=str,
                   help='directory where the WSIs are stored')
group.add_argument('--save_results_fname', type=str, default='save_speed_test_benp_results.csv',
                   help='the file name of the csv file saving the results')
group.add_argument('--load_results_fname', type=str, default='load_speed_test_benp_results.csv',
                   help='the file name of the csv file saving the results')
group.add_argument('--n_wsis', type=int, default=3,
                   help='how many wsis would you like to benchmark')

###########################################################
group = parser.add_argument_group('WSI Arguments')
###########################################################

group.add_argument('--mpp', type=float, default=0.5,
                   help='the mpp at which the patches are going to be stored')
group.add_argument('--patch_size', type=float, default=256,
                   help='the size of the patches to be created')
group.add_argument('--min_tissue_area', type=int, default=1000,
                   help='the minimum tissue area of selected patches')
group.add_argument('--mpp4mask', type=int, default=10,
                   help='idk yet...')
group.add_argument('--max_n_patches', type=int, default=None,
                   help='the maximum number of patches per WSI, if None then all patches are used')
group.add_argument('--load_fmt', type=str, default='numpy',
                   help='the format by which we are loading the WSI patches')

args = parser.parse_args()

###########################################################
# DEFINE THE FUNCTION THAT RUNS ONE WSI
###########################################################

def save_load_speed_run_one_wsi(wsi_fpath:str,
                                mpp:float,
                                patch_size:int,
                                max_n_patches:int,
                                save_dir:str,
                                min_tissue_area:int,
                                mpp4mask:int,
                                fmt:str):
    """

    Parameters
    ----------
    wsi_fpath
    mpp
    patch_size
    max_n_patches
    save_dir
    min_tissue_area
    mpp4mask
    fmt

    Returns the save_results and load_results for that one WSI.
    - save_results['time'] is a dataframe that stores different cumulative save times, eff_n_patches
    - save_results['per_patch'] is a dataframe that stores different per-patch save times, eff_n_patches
    - load_results['time'] is a dataframe that stores different cumulative save times, eff_n_patches
    - load_results['per_patch'] is a dataframe that stores different per-patch save times, eff_n_patches
    - file_sizes return all the file sizes of different storage methods
    -------

    """

    wsi_name, wsi_ext = os.path.splitext(os.path.basename(wsi_fpath))
    patch_coords = make_patch_grid(wsi=wsi_fpath,
                                   patch_size=patch_size,
                                   mpp4patch=mpp,
                                   mpp4mask=mpp4mask,
                                   min_tissue_area=min_tissue_area,
                                   max_n_patches=max_n_patches)

    n_patches = patch_coords.shape[0]

    if args.max_n_patches is None:
        eff_n_patches = n_patches
    else:
        eff_n_patches = min(n_patches, args.max_n_patches)

    save_timer = RuntimeTracker() # initialize the runtime tracker
    save_timer.runtimes['eff_n_patches'] = eff_n_patches

    #########################################
    # INITIALIZING FOLDERSAVER
    #########################################

    folder_savers = [('numpy_arr', NumpySaver()),
                     ('numpy_arr_compressed', NumpySaver(compressed=True)),
                     ('csv', Bag2CsvSaver()),
                     ('parquet', Bag2ParquetSaver()),
                     ('pil_image_png', PilImageSaver('png')),
                     ('pil_image_jpg', PilImageSaver('jpg')),
                     ('torch', TorchSaver())]

    folder_fpaths = {}
    for (kind, saver) in folder_savers:
        folder_fpaths[kind] = os.path.join(save_dir, wsi_name + '__patches' + '-' + kind)



    #########################################
    # SAVE AS H5
    #########################################

    save_fpath_h5 = os.path.join(save_dir, wsi_name + '__patches.h5')

    save_timer.start('h5')
    save_patches_h5(wsi=wsi_fpath,
                    patch_coords=patch_coords,
                    mpp=mpp,
                    patch_size=patch_size,
                    fpath=save_fpath_h5)

    save_timer.finish('h5')



    #########################################
    # SAVE AS A FOLDER OF PATCHES
    #########################################

    imgs_folder_fpath = os.path.join(save_dir, 'wsi' + '__patch_images')

    save_timer.start('folder image patches v1')
    save_patches_folder(wsi=wsi_fpath,
                        patch_coords=patch_coords,
                        mpp=mpp,
                        patch_size=patch_size,
                        folder=imgs_folder_fpath)

    save_timer.finish('folder image patches v1')

    save_timer.start('folder image patches v1, parallel')
    save_patches_folder_par(wsi=wsi_fpath,
                            patch_coords=patch_coords,
                            mpp=mpp,
                            patch_size=patch_size,
                            stub='item-',
                            folder=imgs_folder_fpath,
                            chunksize=10)

    save_timer.finish('folder image patches v1, parallel')



    #########################################
    # SAVE AS A FOLDER OF PATCHES
    #########################################

    for (kind, saver) in folder_savers:

        if kind.startswith('pil'):
            fmt = 'pil'
        elif kind.startswith('torch'):
            fmt = 'torch'
        else:
            fmt = 'numpy'

        if os.path.exists(folder_fpaths[kind]):
            safe_delete_folder(folder_fpaths[kind])

        ###############
        # Non parallel
        ###############

        name = 'folder saver {}'.format(kind)
        save_timer.start(name)
        items = iter_patches_openslide(wsi=wsi_fpath,
                                       patch_coords=patch_coords,
                                       mpp=mpp,
                                       patch_size=patch_size,
                                       fmt=fmt)
        save_in_folder(items=items,
                       saver=saver,
                       path=folder_fpaths[kind],
                       delete_if_exists=True)

        save_timer.finish(name)

        ############
        # Parallel #
        ############

        if os.path.exists(folder_fpaths[kind]):
            safe_delete_folder(folder_fpaths[kind])

        name = 'folder saver {} parallel'.format(kind)
        save_timer.start(name)

        items = iter_patches_openslide(wsi=wsi_fpath,
                                       patch_coords=patch_coords,
                                       mpp=mpp,
                                       patch_size=patch_size,
                                       fmt=fmt)

        save_in_folder_par(items=items,
                           saver=saver,
                           path=folder_fpaths[kind],
                           chunksize=10,
                           delete_if_exists=True)

        save_timer.finish(name)

    save_results = pd.Series(save_timer.runtimes).to_frame('time')
    save_results['per_patch'] = save_results['time'] / eff_n_patches
    save_results.sort_values('time')

    file_sizes = {}

    file_sizes['wsi_svs'] = [get_file_size(wsi_fpath)]
    file_sizes['h5'] = [get_file_size(save_fpath_h5)]

    for kind, path in folder_fpaths.items():
        file_sizes[kind] = [get_folder_size(path)]

    pd.Series(file_sizes).sort_values()




    ##########################################################
    # LOADING LOADING LOADING
    ##########################################################

    load_timer = RuntimeTracker()
    load_timer.runtimes['eff_n_patches'] = eff_n_patches

    #######
    # H5 #
    ######
    load_timer.start('h5')
    load_patches_h5(save_fpath_h5, fmt=fmt)
    load_timer.finish('h5')

    ##########
    # Folder #
    ##########

    load_timer.start('folder images')
    patches_folder = load_patches_folder(imgs_folder_fpath, stub='item-')
    load_timer.finish('folder images')

    #############
    # Openslide #
    #############
    load_timer.start('openslide')
    load_patches_openslide(wsi=wsi_fpath,
                           patch_coords=patch_coords,
                           mpp=mpp,
                           patch_size=patch_size,
                           fmt=fmt)
    load_timer.finish('openslide')

    for (kind, saver) in folder_savers:
        name = 'folder saver {}'.format(kind)

        ###############
        # Non parallel
        ###############
        load_timer.start(name)
        load_from_folder(path=folder_fpaths[kind],
                         saver=saver)

        load_timer.finish(name)

        ############
        # Parallel #
        ############

        name = 'folder saver {} parallel'.format(kind)
        load_timer.start(name)

        load_from_folder_par(path=folder_fpaths[kind],
                             saver=saver,
                             chunksize=10)

        load_timer.finish(name)


    load_timer.start('folder images, ray')
    load_patches_folder_ray(folder=imgs_folder_fpath, stub='item-', fmt=fmt)
    load_timer.finish('folder images, ray')

    for cs in [1, 10, 100]:
        #############
        # Openslide #
        #############
        name = 'openslide parallel, chunksize={}'.format(cs)
        load_timer.start(name)
        load_patches_openslide_par(wsi=wsi_fpath,
                                   patch_coords=patch_coords,
                                   mpp=mpp,
                                   patch_size=patch_size,
                                   fmt=fmt,
                                   use_par=True,
                                   chunksize=cs)
        load_timer.finish(name)

        ###########
        # Folder #
        ##########

        name = 'folder parallel, chunksize={}'.format(cs)
        load_timer.start(name)
        load_patches_folder_par(folder=imgs_folder_fpath,
                                fmt=fmt,
                                chunksize=cs)
        load_timer.finish(name)

    load_results = pd.Series(load_timer.runtimes).to_frame('time')
    load_results['per_patch'] = load_results['time'] / eff_n_patches
    load_results.sort_values('time')

    return save_results, load_results, file_sizes











############################################################################
# THE ACTUAL ITERATIVE SCRIPT
############################################################################

dir_lst = os.listdir(args.wsi_dir)
svs_files = [f for f in dir_lst if os.path.isfile(os.path.join(args.wsi_dir, f)) and f.endswith('.svs')]
save_results_list = []
load_results_list = []
file_sizes_list = []
wsi_root_name_list = []

n = 0
while n < args.n_wsis:
    try:
        wsi_fpath = os.path.join(args.wsi_dir, random.choice(svs_files))
        wsi_root_name, ext = os.path.splitext(os.path.basename(wsi_fpath))
        save_results, load_results, file_sizes = save_load_speed_run_one_wsi(wsi_fpath=wsi_fpath,
                                                                             mpp=args.mpp,
                                                                             mpp4mask=args.mpp4mask,
                                                                             patch_size=args.patch_size,
                                                                             max_n_patches=args.max_n_patches,
                                                                             save_dir=args.save_dir,
                                                                             min_tissue_area=args.min_tissue_area,
                                                                             fmt=args.load_fmt)

        save_results_list.append(save_results)
        load_results_list.append(load_results)
        file_sizes_list.append(file_sizes)
        wsi_root_name_list.append(wsi_root_name)

        n += 1

    except KeyboardInterrupt:
        print("Stopping program...")
        break  # Break out of the loop

# Initialize an empty DataFrame
save_combined_df = pd.DataFrame()

# Iterate over each DataFrame and name pair
for df, name, file_sizes in zip(save_results_list, wsi_root_name_list, file_sizes_list):
    # Keep only the 'per_patch' column and rename it
    df_per_patch = df[['per_patch']].rename(columns={'per_patch': name})
    df_per_patch = df_per_patch.T.add_suffix('...per_patch')

    df_cum = df[['time']].rename(columns={'time': name})
    df_cum = df_cum.T.add_suffix('...cumulative')

    df_sizes = pd.DataFrame(file_sizes)
    df_sizes = df_sizes.rename(index={0: name})
    df_sizes = df_sizes.add_suffix('...file_size')

    df = pd.concat([df_per_patch, df_cum, df_sizes], axis=1) # Combine the per patch, and cumulative time, and patch_size

    # Add the modified DataFrame to the combined DataFrame
    if save_combined_df.empty:
        save_combined_df = df
    else:
        save_combined_df = pd.concat([save_combined_df, df], axis=1)

# Initialize an empty DataFrame
load_combined_df = pd.DataFrame()

for df, name in zip(load_results_list, wsi_root_name_list):
    # Keep only the 'per_patch' column and rename it
    df_per_patch = df[['per_patch']].rename(columns={'per_patch': name})
    df_per_patch = df_per_patch.T.add_suffix('...per_patch')

    df_cum = df[['time']].rename(columns={'time': name})
    df_cum = df_cum.T.add_suffix('...cumulative')

    df = pd.concat([df_per_patch, df_cum], axis=1) # Combine the per patch, and cumulative time, and patch_size

    # Add the modified DataFrame to the combined DataFrame
    if load_combined_df.empty:
        load_combined_df = df
    else:
        load_combined_df = pd.concat([load_combined_df, df], axis=0)

save_results_fpath = os.path.join(args.results_dir, args.save_results_fname)
load_results_fpath = os.path.join(args.results_dir, args.load_results_fname)

save_combined_df.to_csv(save_results_fpath)
load_combined_df.to_csv(load_results_fpath)

