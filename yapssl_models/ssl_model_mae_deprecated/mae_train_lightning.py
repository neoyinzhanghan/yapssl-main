import os
import argparse

import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from ssl_models.ssl_model_mae.mae_lightning import MAELightning
from ssl_models.ssl_models_mae.yaimpl.utils import savefig, parse_n_cpu

# Make sure to install timm 0.4.12
### pip install timm==0.4.12 ###


########################
### ARGUMENT PARSING ###
########################

parser = argparse.\
    ArgumentParser(description='Train an MAE feature extractor',
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

####################################
group = parser.add_argument_group('File paths')
####################################

group.add_argument('--data_path', default='~/Documents/neo/diverse_lusc_patches', type=str,
                   help='folder containing train and test data. Must structured as /data_path/train/label/')

####################################
group = parser.add_argument_group('Training Hyperparameters')
####################################

group.add_argument('--batch_size', default=64, type=int,
                    help='Batch size... duh...')
group.add_argument('--test_batch_size', default=32, type=int,
                    help='Batch size... duh...')
group.add_argument('--epochs', default=800, type=int,
                    help='total number of epochs during training')
group.add_argument('--mask_ratio', default=0.75, type=float,
                    help='Mask ratio of the MAE')
group.add_argument('--blr', default=1e-3, type=float,
                    help='this base learning rate is used to compute the actual learning rate')
group.add_argument('--weight_decay', default=0.05, type=float,
                    help='weight decay used during training')
# group.add_argument('--accum_iter', default=1, type=int,
#                     help='accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
group.add_argument('--max_num_batches', default=None, type=int,
                    help='the maximum number of batches per epoch')
group.add_argument('--warm_up_epochs', default=40, type=int,
                    help='lr linearly scales until the warm-up epochs')

####################################
group = parser.add_argument_group('Hardwares Configuration')
####################################

group.add_argument('--num_workers', default=1, type=int,
                    help='the number of workers in parsing batches')
group.add_argument('--num_accelerators', default=1, type=int,
                    help='the number of accelerators in distributed training')
group.add_argument('--type_accelerator', default='gpu', type=str,
                    help='the type of training accelerator to use')

####################################
group = parser.add_argument_group('Checkpoints & Record Keeping')
####################################

group.add_argument('--save_every_k_epochs', default=20, type=int,
                   help='how frequently to save checkpoints')

args = get_args_parser()








############################
### MAIN TRAINING SCRIPT ###
############################

# DATA TRANSFORMATION
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# DATASET CONFIGURATION
dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
dataset_test = datasets.ImageFolder(os.path.join(args.data_path, 'test'), transform=transform_train)
train_loader = DataLoader(dataset_train,
                          batch_size=args.batch_size,
                          num_workers=parse_n_cpu(args.num_workers))
test_loader = DataLoader(dataset_test,
                         batch_size=args.test_batch_size,
                         num_workers=parse_n_cpu(args.num_workers))



# LEARNING RATE CALIBRATION
eff_batch_size = args.batch_size * args.num_accelerator #* args.accum_iter

lr = args.blr * eff_batch_size / 256


# MODEL SET-UP
autoencoder = MAELightning(lr=lr,
                           warm_up_epochs=args.warm_up_epochs,
                           total_epochs=args.epochs,
                           mask_ratio=args.mask_ratio,
                           weight_decay=args.weight_decay)

# CHECKPOINT CALLBACK SETTING
checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                      every_n_epochs=args.save_every_k_epochs)


# LIGHTNING TRAINER SETTING
trainer = pl.Trainer(max_epochs=args.epochs,
                     limit_train_batches=args.max_num_batches,
                     callbacks=[checkpoint_callback],
                     devices=args.num_accelerator,
                     accelerator=args.type_accelerator)

trainer.fit(autoencoder, train_loader, test_loader)
