import yatt.nn.save_patch_features
import yatt.nn.WsiPatchesDataset
import os
import glob
import functools

from mae.mae_lightning import MAELightning
from yatt.patch import get_full_grid_patch_coords_level0
from yatt.level_info import get_level_info_df

########################
### ARGUMENT PARSING ###
########################

parser = argparse.\
    ArgumentParser(description='Run MAE patch feature extraction from a collection of test WSIs',
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

####################################
group = parser.add_argument_group('WSI file path')
####################################

group.add_argument('--wsi_folder_fpath', type=str, default=None,
                   help='File path to the folder with WSIs from which you intend to extract patch features')
group.add_argument('--save_fpath', type=str, default=None,
                    help='Directory in which you would like to save the output')

####################################
group = parser.add_argument_group('WSI patching specifications')
####################################

group.add_argument('--mpp', default=0.57142857142, type=float,
                    help='The mpp at which you would like the patching to be performed, note that this must match '
                         'with the input size of the patch extraction model.')
group.add_argument('--patch_size', default=224, type=float,
                    help='The patch size.')

####################################
group = parser.add_argument_group('Model specifications')
####################################

group.add_argument('--chkpt_fpath', type=str,
                    help='Path to the MAE checkpoint you intend to use for feature extraction.')

args = parser.parse_args()




######################
### LOAD THE MODEL ###
######################

extraction_model = MAELightning.from_check_point(args.chkpt_fpath)
extraction_model.eval()
forward_encoder_func = functools.partial(extraction_model.forward_encoder, extraction_model)

def hook_flatten_latent_feature(input_images):
    # Call the model_func with the input_images
    encoder_output, _, _ = forward_encoder_func(input_images, mask_ratio=0)

    batch_size = encoder_output.size(0)
    flattened_encoder_output = encoder_output.reshape(batch_size, -1)

    return flattened_encoder_output


####################################
### LOAD, PATCH, EXTRACT THE WSI ###
####################################

def _save_patch_features_front_end(wsi_fpath):
    wsi = open_slide(wsi_fpath)
    level_info = get_level_info_df(wsi=wsi)
    coords_level0 = get_full_grid_patch_coords_level0(patch_size=args.patch_size,
                                                      mpp=args.mpp,
                                                      level_info=level_info)

    wsi_patches_dataset = WsiPatchesDataset(wsi_fpath=args.wsi_fpath,
                                            coords_level0=coords_level0,
                                            patch_size=args.patch_size,
                                            mpp=args.mpp)

    save_patch_features(patches_dataset=wsi_patches_dataset,
                        model=hook_flatten_latent_feature,
                        fpath=args.save_fpath)


###################################################
### LOOP OVER ALL SVS FILES IN wsi_folder_fpath ###
###################################################

svs_files = glob.glob(os.path.join(wsi_folder_fpath, "*.svs"))

for wsi_fpath in svs_files:
    _save_patch_features_front_end(wsi_fpath)

