import os
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump
import torch
from tqdm import tqdm

import argparse

from torchvision.transforms import ToTensor

from openslide import open_slide

from yatt.MaskAndPatch import MaskAndPatch
from yatt.tissue_mask import RgbTissueMasker
from yatt.nn.WsiPatchesDataset import WsiPatchesDataset

from yatt.nn.extract_patch_features import save_patch_features, H5FeatureSaver
from yatt.nn.vision_models import load_model, append_flattened_mean_pool
from yatt.file_utils import join_and_make
from yatt.viz import savefig
from yatt.level_info import get_level_info_df
from yatt.wsi_extensions import find_wsi_paths

from yapssl_models.ssl_eval import PatchFeatureExtractor

# Make sure to install timm 0.4.12 if using MAE feature extractor
### pip install timm==0.4.12 ###




####################################
# ARGUMENT PARSING
####################################

parser = argparse.ArgumentParser()

####################################
group = parser.add_argument_group('Dataset and paths')
####################################

group.add_argument('--data_dir', type=str, nargs='+',
                   help='Directory or directories containing WSIs ')

group.add_argument('--top_save_dir', type=str,
                   help='Directory where to save the output. The feautres will be saved in save_dir/patch_features/.')


####################################
group = parser.add_argument_group('Patch arguments')
####################################

group.add_argument('--patch_size', default=224, type=int,
                   help='Size of the patch in pixels to extract at the specified mpp.')


group.add_argument('--mpp4patch', default=0.57142857142, type=float,
                   help='Microns per pixel resolution for the image patches. The default of 0.5 is about 20x (200x effective) magnification.')

group.add_argument('--min_tissue_area', default=1000, type=float,
                   help='Only include patches with at least this much tissue in microns squared.')


group.add_argument('--existing_patch_dir', default=None, type=str,
                   help='Instead of computing the patch grid we can load it from an already obtained patch grid (e.g. so we can extract multiple sets of featurs from the same patches). This should be the path to the directory containing the already compted patch grids.')

group.add_argument('--max_n_patches', default=None, type=int,
                   help='Cap the number of patches -- mainly for degugging/prototyping. None means no capping.')

####################################
group = parser.add_argument_group('Model')
####################################

group.add_argument('--model_name', default='ssl', type=str,
                   help='Which model to use.')

group.add_argument('--ssl_chkpt_fpath', type=str,
                   help='The directory to the pretrained SSL checkpoint feature-extractor')

group.add_argument('--ssl_arch', default='simclr', type=str,
                   help='The SSL architecture that is used for extracting feature.')

####################################
group = parser.add_argument_group('Bureaucracy')
####################################

group.add_argument('--device', default='auto', type=str,
                   help='Which device to use.')

group.add_argument('--num_workers', default=1, type=int,
                   help='Number of workers for the data loader.')

group.add_argument('--batch_size', default=1, type=int,
                   help='Batch size to use.')

group.add_argument('--verbosity', default=1, type=int,
                   help='Verbosity level.')

group.add_argument('--continue_on_error', type=bool, default=False,
                   help='Continue processing even if there is an error.')

args = parser.parse_args()





####################################
# Misc.
####################################

save_patch_grid = True
patch_size = (args.patch_size, args.patch_size)
min_tissue_area = args.min_tissue_area

################
# Setup  paths #
################

dirs = {'feats': join_and_make(args.top_save_dir, 'patch_features')}
if save_patch_grid:
    dirs['patch_viz'] = join_and_make(args.top_save_dir, 'patch_grid_viz')
    dirs['patch_info'] = join_and_make(args.top_save_dir, 'patch_grid_info')


##################
# Load the model #
##################

if args.model_name == 'clam_resnet':
    model = load_model(args.model_name, pretrained = True)

if args.model_name == 'ssl':
    model = PatchFeatureExtractor(chkpt_fpath=args.ssl_chkpt_fpath,
                                  ssl_arch=args.ssl_arch)

if args.model_name != 'clam_resnet':
    if hasattr(model, 'features'):
        model = append_flattened_mean_pool(model.features)

if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.device == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device

model.to(device)

####################################################
# Determine WSIs whose features we want to compute #
####################################################

avail_wsi_fpaths = find_wsi_paths(args.data_dir)
print("Found {} WSIs".format(len(avail_wsi_fpaths)))


for wsi_fpath in tqdm(avail_wsi_fpaths, desc="WSI patch feature extraction"):

    try:
        # wsi_fpath = avail_wsi_fpaths[0]

        wsi_name = Path(wsi_fpath).stem

        ######################
        # Create patch grid #
        #####################
        wsi = open_slide(wsi_fpath)

        patcher = MaskAndPatch(mpp4mask=10,
                            tissue_masker=RgbTissueMasker(),
                            mpp4patch=args.mpp4patch,
                            patch_size=patch_size,
                            store_mask=True
                            )
        patcher.fit_mask_and_patch_grid(wsi)
        patch_df = patcher.get_patch_df()

        # Only include patches with enough tissue
        idxs_tissue = patch_df.query("tissue_area >= @min_tissue_area").index.values
        patch_df['include'] = False
        patch_df.loc[idxs_tissue, 'include'] = True

        print("{}/{} patches ({:1.2f}%) have at least {} microns^2 tissue area".
            format(len(idxs_tissue),
                    patch_df.shape[0],
                    100*len(idxs_tissue)/patch_df.shape[0],
                    args.min_tissue_area))

        ###################
        # Save patch info #
        ###################
        if save_patch_grid:

            # Visualize mask + patch grid
            plt.figure(figsize=(20, 10))
            patcher.plot_mask_and_grid(min_tissue_area=args.min_tissue_area, thickness=2)
            savefig(os.path.join(dirs['patch_viz'], '{}.png'.format(wsi_name)))

            # save patch information
            patch_data = {'patch_df': patch_df,
                        'mpp': args.mpp4patch,
                        'mpp_level0': get_level_info_df(wsi).loc[0, 'mpp'],
                        'patch_size': patch_size
                        }

            dump(patch_data, os.path.join(dirs['patch_info'], wsi_name))

        # subset to only the patches we will include
        patch_df = patch_df.query('include')
        if args.max_n_patches is not None:
            patch_df = patch_df.iloc[0:args.max_n_patches]

        #########################
        # Setup patches dataset #
        #########################

        coords_level0 = patch_df[['x', 'y']].values
        dataset = WsiPatchesDataset(wsi_fpath=wsi_fpath,
                                    coords_level0=coords_level0,
                                    patch_size=patch_size,
                                    mpp=args.mpp4patch,
                                    transform=ToTensor())


        #############################
        # Extract and save features #
        #############################
        feats_fpath = os.path.join(dirs['feats'], '{}.h5'.format(wsi_name))

        feature_saver = H5FeatureSaver(patch_identif=patch_df.index.values,
                                    identif_name='patch_idx',
                                    fpath=feats_fpath,)
        
        batch_size = save_patch_features(patches_dataset=dataset,
                                        model=model,
                                        feature_saver=feature_saver,
                                        batch_size=args.batch_size,
                                        loader_kws={'num_workers': args.num_workers},
                                        device=device,
                                        verbosity=args.verbosity)
    except Exception as e:
        print("Error with {}".format(wsi_fpath))
        print(e)
        if args.continue_on_error:
            continue 
    
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        break
