import argparse
import os
import random
import torch
from openslide import open_slide
from tqdm import tqdm
import torchvision.transforms as transforms

from yatt.MaskAndPatch import MaskAndPatch
from yatt.tissue_mask import RgbTissueMasker
from yatt.read_image import get_patch_at_mpp

#########################################################
# ARGUMENT PARSING
#########################################################

parser = argparse.ArgumentParser()


#########################################################
group = parser.add_argument_group('Directories')
#########################################################

group.add_argument('--wsi_dir', type=str,
                    help='Directory in which the WSIs that you want to patch are stored')

group.add_argument('--save_destination', type=str,
                    help='Directory in which you would like to store the resulting patches')


#########################################################
group = parser.add_argument_group('WSI Hyperparameters')
#########################################################

parser.add_argument('--n_wsis', default=100, type=int,
                    help='The number of wsis from the directory that you intend to patch (default=100)')

parser.add_argument('--max_num_patches_per_wsi', default=None, type=int,
                    help='How many tissue patches would you like to extract out of each wsi? (default=1000)')

parser.add_argument('--mpp', default=0.57142857142, type=float,
                    help='The mpp at which you would like the patching to be performed')

parser.add_argument('--patch_size', default=224, type=int,
                    help='The size of the patches')

parser.add_argument('--min_tissue_area', default=1000, type=int,
                    help='The minimum tissue area of selected patches')

parser.add_argument('--patch_ext', default='.png', type=str,
                    help='The file extension under which the patches are to be saved')

args = parser.parse_args()


#########################################################
# DEFINE PATCH AND DOWNLOAD FUNCTION FOR ONE WSI
#########################################################

def _patch_and_download(wsi_dir,
                        wsi_name,
                        save_destination,
                        tag=''):

    wsi_fpath = os.path.join(wsi_dir, wsi_name)
    root = os.path.splitext(os.path.basename(wsi_name))[0]
    wsi = open_slide(wsi_fpath)
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

    # subset to only the patches we will include
    patch_df_included = patch_df.query('include')
    patch_coords = patch_df_included[['x', 'y']].values

    if args.max_num_patches_per_wsi is None:
        total = len(patch_coords)
    else:
        total = min(len(patch_coords), args.max_num_patches_per_wsi)

    for idx in tqdm(range(total), desc='Saving patches from --- ' + root + tag, position=1):
        coords_level0 = patch_coords[idx]

        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=args.mpp,
                                 coords_level0=coords_level0,
                                 patch_size_mpp=args.patch_size,
                                 # tol=self.tol
                                 )

        if args.patch_ext == '.pt':
            to_tensor = transforms.ToTensor()
            patch_tensor = to_tensor(patch)

            file_name = 'patch_' + str(idx) + '_' + root + args.patch_ext
            file_path = os.path.join(save_destination, file_name)
            torch.save(patch_tensor, file_path)
        elif args.patch_ext == '.png':
            file_name = 'patch_' + str(idx) + '_' + root + args.patch_ext
            file_path = os.path.join(save_destination, file_name)

            patch.save(file_path)
        else:
            raise ValueError(f'Extension {args.patch_ext} not supported.')


#########################################################
# DEFINE PATCH AND DOWNLOAD FOR ONE WSI
#########################################################

# Traverse through each file in the directory
dir_lst = os.listdir(args.wsi_dir)

wsi_lst = [f for f in dir_lst if os.path.isfile(os.path.join(args.wsi_dir, f)) and f.endswith('.svs')]

already_tried = []

i = 0
while i < args.n_wsis:
    try:
        filename = random.choice(wsi_lst)
        while filename in already_tried:
            filename = random.choice(wsi_lst)
        already_tried.append(filename)

        if os.path.isfile(os.path.join(args.wsi_dir, filename)):
            _patch_and_download(wsi_dir=args.wsi_dir,
                                wsi_name=filename,
                                save_destination=args.save_destination,
                                tag=' --- ' + str(i + 1) + '/' + str(args.n_wsis))
            i += 1

    except ValueError:
        print(f'A value error occurred. Most likely cause is that your specified file extention {args.patch_ext} is not supported. Supported file extension include .png and .pt')
        break

    except Exception:
        print('Some WSI error occurred, continuing with new WSI')
        pass

    except KeyboardInterrupt:
        print('Interrupted by user.')
        break