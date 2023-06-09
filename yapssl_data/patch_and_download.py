import os
import torch
from openslide import open_slide
from tqdm import tqdm
import torchvision.transforms as transforms

from yatt.MaskAndPatch import MaskAndPatch
from yatt.tissue_mask import RgbTissueMasker
from yatt.read_image import get_patch_at_mpp


#########################################################
# DEFINE PATCH AND DOWNLOAD FUNCTION FOR ONE WSI
#########################################################

def _patch_and_download(wsi_dir,
                        wsi_name,
                        save_destination,
                        mpp,
                        patch_size,
                        min_tissue_area,
                        max_num_patches_per_wsi,
                        patch_ext = '.pt',
                        tag=''):

    wsi_fpath = os.path.join(wsi_dir, wsi_name)
    root = os.path.splitext(os.path.basename(wsi_name))[0]
    wsi = open_slide(wsi_fpath)
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

    if max_num_patches_per_wsi is None:
        total = len(patch_coords)
    else:
        total = min(len(patch_coords), max_num_patches_per_wsi)

    for idx in tqdm(range(total), desc='Saving patches from --- ' + root + tag, position=1):
        coords_level0 = patch_coords[idx]

        patch = get_patch_at_mpp(wsi=wsi,
                                 mpp=mpp,
                                 coords_level0=coords_level0,
                                 patch_size_mpp=patch_size,
                                 # tol=self.tol
                                 )

        if patch_ext == '.pt':
            to_tensor = transforms.ToTensor()
            patch_tensor = to_tensor(patch)

            file_name = 'patch_' + str(idx) + '_' + root + patch_ext
            file_path = os.path.join(save_destination, file_name)
            torch.save(patch_tensor, file_path)
        elif patch_ext == '.png':
            file_name = 'patch_' + str(idx) + '_' + root + patch_ext
            file_path = os.path.join(save_destination, file_name)

            patch.save(file_path)
        else:
            raise ValueError(f'Extension {patch_ext} not supported.')