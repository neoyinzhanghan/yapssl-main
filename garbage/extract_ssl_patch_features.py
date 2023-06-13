import os
from pathlib import Path
from joblib import dump, load
import torch
from tqdm import tqdm
import argparse
import pandas as pd
from pprint import pprint
from datetime import datetime
import sys

from torchvision.transforms import ToTensor
from openslide import open_slide

import matplotlib
matplotlib.use('Agg')  # this avoids a very annoying error on
# linux that I don't understand
import matplotlib.pyplot as plt

from yatt.MaskAndPatch import MaskAndPatch
from yatt.tissue_mask import RgbTissueMasker
from yatt.nn.WsiPatchesDataset import WsiPatchesDataset

from yatt.nn.extract_patch_features import save_patch_features, H5FeatureSaver
from yatt.nn.vision_models import load_model, append_flattened_mean_pool
from yatt.file_utils import join_and_make
from yatt.viz import savefig
from yatt.utils import get_traceback
from yatt.level_info import get_level_info_df
from yatt.wsi_extensions import find_wsi_paths
from yatt.RuntimeTracker import RuntimeTracker

from yapssl_models.ssl_eval import PatchFeatureExtractor

# Make sure to install timm 0.4.12 if using MAE feature extractor
### pip install timm==0.4.12 ###

parser = argparse.ArgumentParser()


####################################
group = parser.add_argument_group('Dataset and paths')
####################################

group.add_argument('--data_dir', type=str, nargs='+',
                   help='Directory or directories containing WSIs ')

group.add_argument('--save_dir', type=str,
                   help='Directory where to save the output. The feautres will be saved in save_dir/patch_features/.')


group.add_argument('--skip_wsi_with_feats',
                   action='store_true', default=False,
                   help='Skip a WSI if we already obtained features for it. We know a WSI has features if it is listed in the extraction_info.csv file. This option allows us to resume jobs that have been interupted. It handles things appropriately if features were only partially computed for a wsi.')

# TODO: maybe implement?
group.add_argument('--continue_on_error',
                   action='store_true', default=False,
                   help='Continue running of there is an error.')


####################################
group = parser.add_argument_group('Patch arguments')
####################################

group.add_argument('--patch_size', default=224, type=int,
                   help='Size of the patch in pixels to extract at the specified mpp.')


group.add_argument('--mpp', default=0.57142857142, type=float,
                   help='Microns per pixel resolution for the image patches. The default of 0.5 is about 20x (200x effective) magnification.')

group.add_argument('--min_tissue_area', default=10000, type=float,
                   help='Only include patches with at least this much tissue in microns squared.')


group.add_argument('--existing_patch_dir', default=None, type=str,
                   help='Instead of computing the patch grid we can load it from an already obtained patch grid (e.g. so we can extract multiple sets of featurs from the same patches). This should be the path to the directory containing the already compted patch grids.')

####################################
group = parser.add_argument_group('Model')
####################################
group.add_argument('--model_name', default='ssl', type=str,
                   help='Which model to use.')

group.add_argument('--ssl_chkpt_fpath', type=str,
                   help='The directory to the pretrained SSL checkpoint feature-extractor')

group.add_argument('--ssl_arch', default='mae', type=str,
                   help='The SSL architecture that is used for extracting feature.')

####################################
group = parser.add_argument_group('Bureaucracy')
####################################

group.add_argument('--device', default='auto', type=str,
                   help='Which device to use.')

group.add_argument('--num_workers', default=0, type=int,
                   help='Number of workders for the data loader.')

group.add_argument('--batch_size', default=16, type=int,
                   help='Batch size to use.')

group.add_argument('--use_dp',
                   action='store_true', default=False,
                   help='Use DataParallel.')

group.add_argument('--auto_batch_size',
                   action='store_true', default=False,
                   help='Automatically determinie the batch size.')

####################################
group = parser.add_argument_group('Limit number of WSIs')
####################################

group.add_argument('--max_n_wsi', default=None, type=int,
                   help='Maximum number of WSIs. Useful for prototyping/debugging.')

group.add_argument('--mini',
                   action='store_true', default=False,
                   help='Mini experiment for debugging e.g. just extract a few features.')

args = parser.parse_args()

print(args)
# mpp4patch = 0.5
# patch_size = (256, 256)
# min_area = 1000  # amount of tissue in micron^2 for us to include a patch

# model_name = 'clam_resnet'


# data_dir = '/Users/iaincarmichael/Dropbox/Research/comp_onc/data/tcga/brca/wsi'
# top_save_dir = '/Users/iaincarmichael/Dropbox/Research/comp_onc/projects/gi_screen/notebook/temp_patch_feats'


# save patch grid if we arent loading an existing patch grid
save_patch_grid = args.existing_patch_dir is None

dtype = None  # TODO: decide if we want this

# device = None
# batch_size = 16
# num_workers = 0

# # cap the number of patches -- mainly for degugging/prototyping
# max_n_patches = 10
# man_n_wsis = 10

################
# Setup  paths #
################

dirs = {'feats': join_and_make(args.save_dir, 'patch_features')}
if save_patch_grid:
    dirs['patch_viz'] = join_and_make(args.save_dir, 'patch_grid_viz')
    dirs['patch_info'] = join_and_make(args.save_dir, 'patch_grid_info')

extraction_info_fpath = os.path.join(args.save_dir, 'extraction_info.csv')
error_info_fpath = os.path.join(args.save_dir, 'error_info.csv')


##################
# Load the model #
##################
if args.model_name == 'clam_resnet':
    model = load_model(args.model_name, pretrained=True)

if args.model_name == 'ssl':
    model = PatchFeatureExtractor(chkpt_fpath=args.ssl_chkpt_fpath,
                                  ssl_arch=args.ssl_arch)

if args.model_name != 'clam_resnet':
    if hasattr(model, 'features'):
        model = append_flattened_mean_pool(model.features)

if args.device == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device

####################################################
# Determine WSIs whose features we want to compute #
####################################################

avail_wsi_fpaths = find_wsi_paths(args.data_dir, recursive=False)
print("Found {} WSIs".format(len(avail_wsi_fpaths)))


# (maybe) Subset to the WSIs for whom we do not already have features
# as determined by reading the extraction_info.csv file
if os.path.exists(extraction_info_fpath):

    if args.skip_wsi_with_feats:
        # find WSIs with features that already finished computing and dorp them
        extraction_info = pd.read_csv(extraction_info_fpath)
        wsis_with_feats = set(extraction_info['wsi'])

        avail_wsi_fpaths = [fpath for fpath in avail_wsi_fpaths if
                            Path(fpath).stem not in wsis_with_feats]

        print("Limiting to {} WSIs that do not already have features".
              format(len(avail_wsi_fpaths)))

    else:
        # we want to overwrite the exiting features so delete
        # the existing extraction_info_fpath
        os.remove(extraction_info_fpath)


# Maybe subset fpaths
if args.mini:
    args.max_n_wsi = 2
if args.max_n_wsi is not None:
    avail_wsi_fpaths = avail_wsi_fpaths[0:args.max_n_wsi]

####################################
# Extract features for WSI patches
####################################

for wsi_fpath in tqdm(avail_wsi_fpaths,
                      desc="WSI patch feature extraction"):

    try:
        wsi_name = Path(wsi_fpath).stem
        wsi = open_slide(wsi_fpath)

        info = {'wsi': wsi_name}

        rtt = RuntimeTracker()
        rtt.start('overall_runtime')

        print("\n\nExtracting features for {}".format(wsi_name))

        ######################
        # Create patch grid #
        #####################

        if args.existing_patch_dir is None:

            # creat patch gride
            patcher = MaskAndPatch(mpp4mask=10,
                                   tissue_masker=RgbTissueMasker(),
                                   mpp4patch=args.mpp,
                                   patch_size=args.patch_size,
                                   store_mask=True
                                   )

            rtt.start('patch_runtime')
            patcher.fit_mask_and_patch_grid(wsi)
            rtt.finish('patch_runtime')

        else:

            # load patch grid from already computed one
            patcher_fpath = os.path.join(args.existing_patch_dir,
                                         wsi_name + '__patch_grid')

            patcher = load(patcher_fpath)

        patch_df = patcher.get_patch_df()

        # Only include patches with enough tissue
        idxs_tissue = patch_df.query("tissue_area >= {}".
                                     format(args.min_tissue_area)).index.values
        patch_df['include'] = False
        patch_df.loc[idxs_tissue, 'include'] = True

        info['n_tissue_patches'] = len(idxs_tissue)
        info['n_total_patches'] = patch_df.shape[0]

        print("{}/{} patches ({:1.2f}%) have at "
              "least {} microns^2 tissue area".
              format(info['n_tissue_patches'],
                     info['n_total_patches'],
                     100*info['n_tissue_patches']/info['n_total_patches'],
                     args.min_tissue_area))

        # subset to only the patches we will include
        patch_df = patch_df.query('include')

        # maybe do mini
        if args.mini:
            max_n_patches = 10
            patch_df = patch_df.iloc[0:max_n_patches]

        ###################
        # Save patch info #
        ###################
        if save_patch_grid:

            # Visualize mask + patch grid
            plt.figure(figsize=(20, 10))
            patcher.\
                plot_mask_and_grid(min_tissue_area=args.min_tissue_area,
                                   thickness=2)
            savefig(os.path.join(dirs['patch_viz'], '{}.png'.
                    format(wsi_name)))

            # # save patch information
            # patch_data = {'patch_df': patch_df,
            #               'mpp': args.mpp,
            #               'mpp_level0': get_level_info_df(wsi).loc[0, 'mpp'],
            #               'patch_size': args.patch_size
            #               }
            # dump(patch_data, os.path.join(dirs['patch_info'], wsi_name))
            dump(patcher,  os.path.join(dirs['patch_info'],
                                        wsi_name + '__patch_grid'))
        #########################
        # Setup patches dataset #
        #########################

        coords_level0 = patch_df[['x', 'y']].values
        dataset = WsiPatchesDataset(wsi_fpath=wsi_fpath,
                                    coords_level0=coords_level0,
                                    patch_size=args.patch_size,
                                    mpp=args.mpp,
                                    transform=ToTensor())

        #############################
        # Extract and save features #
        #############################
        loader_kws = {'num_workers': args.num_workers}

        feats_fpath = os.path.join(dirs['feats'], '{}.h5'.format(wsi_name))

        feature_saver = H5FeatureSaver(fpath=feats_fpath,
                                       patch_identif=patch_df.index.values,
                                       identif_name='patch_idx',
                                       delete_if_exists=True)

        rtt.start('feat_extract_runtime')

        batch_info = save_patch_features(patches_dataset=dataset,
                                         model=model,
                                         feature_saver=feature_saver,
                                         batch_size=args.batch_size,
                                         use_dp=args.use_dp,
                                         loader_kws=loader_kws,
                                         device=device,
                                         dtype=dtype,
                                         verbosity=2)
        rtt.finish('feat_extract_runtime')

        ##########################
        # Handle extraction info #
        ##########################
        rtt.finish('overall_runtime')

        info.update(**rtt.runtimes)

        batch_info_totals = pd.DataFrame(batch_info).sum(axis=0).to_dict()
        inst_avgs_runtimes = {k: v / batch_info_totals['size']
                              for (k, v) in batch_info_totals.items()
                              if 'runtime' in k}
        # batch_info_totals = {'batch_total-{}'.format(k): v
        #                      for (k, v) in batch_info_totals.items()}

        info.update(**rtt.runtimes)
        info.update(**{'batch_total-{}'.format(k): v
                       for (k, v) in batch_info_totals.items()})
        info['finsh_time'] = datetime.now().strftime('%Y-%m-%d__%H:%M:%S')
        pprint(info)
        print("avg instance time", inst_avgs_runtimes)

        pd.DataFrame([info]).\
            to_csv(extraction_info_fpath,
                   mode='a',
                   index=False,
                   header=not os.path.exists(extraction_info_fpath))

    except KeyboardInterrupt:
        # User abort, exit like normal
        sys.exit()
        pass

    except Exception as e:

        if args.continue_on_error:
            ######################################
            # Save the error info an continue on #
            ######################################
            traceback = get_traceback(e)
            error_name = type(e).__name__
            print("{} exception thrown in call with kws:\n{}".
                  format(error_name, traceback))

            error_info = {'wsi': wsi_name,
                          'error_name': error_name,
                          'traceback': traceback}

            pd.DataFrame([error_info]).\
                to_csv(error_info_fpath,
                       mode='a',
                       index=False,
                       header=not os.path.exists(error_info_fpath))

        else:
            raise e
