import argparse
import os
import random
from yapssl_data.patch_and_download import _patch_and_download

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

parser.add_argument('--n_wsis', default=None, type=int,
                    help='The number of wsis from the directory that you intend to patch (default=None), if None, then all images will be used.')

parser.add_argument('--max_num_patches_per_wsi', default=None, type=int,
                    help='How many tissue patches would you like to extract out of each wsi? (default=1000)')

parser.add_argument('--mpp', default=0.57142857142, type=float,
                    help='The mpp at which you would like the patching to be performed')

parser.add_argument('--patch_size', default=224, type=int,
                    help='The size of the patches')

parser.add_argument('--min_tissue_area', default=1000, type=int,
                    help='The minimum tissue area of selected patches')

args = parser.parse_args()







#########################################################
# PATCHING SCRIPT
#########################################################

if __name__ == '__main__':

    if args.n_wsis is None:

        dir_lst = os.listdir(args.wsi_dir)
        wsi_lst = [f for f in dir_lst if os.path.isfile(os.path.join(args.wsi_dir, f)) and f.endswith('.svs')]

        for i in range(len(wsi_lst)):
            try:
                filename = wsi_lst[i]
                if os.path.isfile(os.path.join(args.wsi_dir, filename)):

                    wsi_folder_name = os.path.splitext(os.path.basename(filename))[0]
                    wsi_folder_path = os.path.join(args.save_destination, wsi_folder_name)

                    _patch_and_download(wsi_dir=args.wsi_dir,
                                        wsi_name=filename,
                                        save_destination=wsi_folder_path,
                                        mpp=args.mpp,
                                        patch_size=args.patch_size,
                                        min_tissue_area=args.min_tissue_area,
                                        max_num_patches_per_wsi=args.max_num_patches_per_wsi,
                                        tag=' --- ' + str(i + 1) + '/' + str(args.n_wsis))

            except Exception:
                print('Some WSI error occurred, continuing with new WSI')
                pass

            except KeyboardInterrupt:
                print('Interrupted by user.')
                break

    else:
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
                    # first create a folder named after the WSI

                    wsi_folder_name = os.path.splitext(os.path.basename(filename))[0]
                    wsi_folder_path = os.path.join(args.save_destination, wsi_folder_name)

                    _patch_and_download(wsi_dir=args.wsi_dir,
                                        wsi_name=filename,
                                        save_destination=wsi_folder_path,
                                        mpp=args.mpp,
                                        patch_size=args.patch_size,
                                        min_tissue_area=args.min_tissue_area,
                                        max_num_patches_per_wsi=args.max_num_patches_per_wsi,
                                        tag=' --- ' + str(i + 1) + '/' + str(args.n_wsis))
                    i += 1


            except Exception:
                print('Some WSI error occurred, continuing with new WSI')
                pass

            except KeyboardInterrupt:
                print('Interrupted by user.')
                break