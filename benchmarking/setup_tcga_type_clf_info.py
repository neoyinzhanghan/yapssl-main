import os
import argparse
import json
import pandas as pd
from glob import glob

from yatt.tcga.utils import load_tcga_cdr
from yaimpl.data_split import add_splts_to_label_df
from wsip.tcga.case_wsi import get_tcga_case_to_wsi_map

parser = argparse.\
    ArgumentParser(description='Trains a classification model.')

parser.add_argument('--save_dir', type=str,
                    help='Directory where to save config files.')

parser.add_argument('--wsi_dirs', type=str, nargs='+',
                    help='Directories containing the whole slide images or whole slide image features.')

parser.add_argument('--cdr_fpath', type=str, default='yapssl_resources/TCGA/mmc1.xlsx',
                    help='Path to TCGA clean data resource i.e. mmc1.xlsx file.')

parser.add_argument('--label_col', type=str, default='type',
                    help='Which column of the CDR to use as the class labels.')

args = parser.parse_args()

# which column to pull from metadata to use as a label
drop_wsi_if_not_in_metadata = False


###############
# Setup paths #
###############

case2wsi_fpath = os.path.join(args.save_dir, 'case2wsi.json')
case_labels_fpath = os.path.join(args.save_dir, 'case_labels.csv')
case_metadata_fpath = os.path.join(args.save_dir, 'case_metadata.csv')


#############################
# Find available WSIs/cases #
#############################

# Find all WSIs
# wsi_fpaths = find_wsi_paths(folder=wsi_dir, recursive=True)
wsi_fpaths = []
for d in args.wsi_dirs:
    fpaths_this_dir = glob(os.path.join(d, 'TCGA*'))
    print("{} TCGA WSIs found in {}".format(len(fpaths_this_dir), d))
    wsi_fpaths.extend(fpaths_this_dir)

# map cases to WSIs
case_to_wsi_map = get_tcga_case_to_wsi_map(wsi_fpaths=wsi_fpaths)

# load the case metadata for the available cases
case_metadata = load_tcga_cdr(args.cdr_fpath)
# TODO: give option to process

# compare available cases to cases in metadata file
if drop_wsi_if_not_in_metadata:
    case_to_wsi_map = {case: wsis for (case, wsis) in case_to_wsi_map.items()
                       if case in case_metadata.index}
else:
    assert all([case in case_metadata.index for case in case_to_wsi_map])


case_metadata = case_metadata.loc[list(case_to_wsi_map.keys())]


########################
# Create case label df #
########################
case_labels = case_metadata[args.label_col].values
case_labels = pd.DataFrame({'case_name': case_metadata.index.values,
                            'label': case_labels})

# Trian/val/test split
# case_labels['split'] = 'train'
case_labels = add_splts_to_label_df(case_labels,
                                    train_size=0.8,
                                    val_size=0.1,
                                    test_size=0.1,
                                    stratify=True,
                                    random_state=1,
                                    name_col='case_name')

##############
# Save info #
##############
os.makedirs(args.save_dir, exist_ok=True)

# save case WSI map
with open(case2wsi_fpath, "w") as file:
    json.dump(case_to_wsi_map, file, indent=0)

# save case labels + metadata
case_labels.to_csv(case_labels_fpath, index=False)
case_metadata.to_csv(case_metadata_fpath)
