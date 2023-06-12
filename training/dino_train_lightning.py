import os
import argparse

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torchmetrics import MetricCollection
from lightly.data import LightlyDataset

import pytorch_lightning as pl
from lightly.data.multi_view_collate import MultiViewCollate
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset
from lightly.transforms.simclr_transform import SimCLRTransform
from yapssl_models.simclr_lightning import SimCLR
# from ssl_models.utils.grab_pt_patches import MySSLDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from yaimpl.utils import parse_n_cpu


########################
### ARGUMENT PARSING ###
########################

parser = argparse.ArgumentParser()

####################################
group = parser.add_argument_group('File paths')
####################################

group.add_argument('--data_path', type=str,
                   help='folder containing train and test data. Must structured as /data_path/train/label/')

####################################
group = parser.add_argument_group('Training Hyperparameters')
####################################

group.add_argument('--batch_size', default=128, type=int, #1024 is the paper default inner_eff_batch_size
                    help='Batch size for training')
group.add_argument('--test_batch_size', default=64, type=int,
                    help='Batch size for testing')
group.add_argument('--epochs', default=300, type=int,
                    help='total number of epochs during training')
group.add_argument('--blr', default=0.0005, type=float,
                    help='this base learning rate is used to compute the actual learning rate')
group.add_argument('--max_num_batches', default=None, type=int,
                    help='the maximum number of batches per epoch')
group.add_argument('--weight_decay', default=0.05, type=float,
                    help='In deep learning, weight decay is a regularization technique used to prevent overfitting by adding a penalty to the loss function, which effectively reduces the magnitude of the weights in the model.')
group.add_argument('--warm_up_epochs', default=40, type=int,
                    help='the number of epochs for a linear training warm-up')
group.add_argument('--accum_grad', default=16, type=int,
                    help='accumulate gradients for accum_grad batches')
group.add_argument('--min_lr', type=float, default=1e-6, 
                   help="Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.")
group.add_argument('--sub_patch_size', type=int, default=8,
                   help="Sub patch size for the DINO feature extraction.")

####################################
group = parser.add_argument_group('DINO Hyperparameters')
####################################

####################################
group = parser.add_argument_group('Hardwares Configuration')
####################################

group.add_argument('--num_workers', default=1, type=int,
                    help='the number of workers in parsing batches')
group.add_argument('--num_accelerators', default=1, type=int,
                    help='the number of accelerators in distributed training') # the paper default is 16, but I reduce it to 8
group.add_argument('--type_accelerator', default='gpu', type=str,
                    help='the type of training accelerator to use')

####################################
group = parser.add_argument_group('Checkpoints & Record Keeping')
####################################

group.add_argument('--top_k_epochs', default=20, type=int,
                   help='to save the best k checkpoints measured by lowest validation loss')


args = parser.parse_args()









if __name__ == '__main__':

    ############################
    ### MAIN TRAINING SCRIPT ###
    ############################

    # LEARNING RATE COMPUTATION

    inner_eff_batch_size = args.batch_size * args.accum_grad
    lr = args.blr * inner_eff_batch_size*args.num_accelerators/256

    # DATA AUGMENTATION

    transform = DINOTransform()
    dataset_train = LightlyDataset(os.path.join(os.path.expanduser(args.data_path),'train'), transform=transform)
    dataset_test = LightlyDataset(os.path.join(os.path.expanduser(args.data_path),'test'), transform=transform)

    # DATALOADERS

    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=parse_n_cpu(args.num_workers))

    val_loader = DataLoader(dataset_test,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=parse_n_cpu(args.num_workers))



    model = DINO(lr=lr, 
                 min_lr=args.min_lr, 
                 epochs=args.epochs, 
                 niter_per_ep=len(train_loader), 
                 sub_patch_size=args.sub_patch_size,)

    # CHECKPOINT CALLBACK SETTING
    checkpoint_callback = ModelCheckpoint(save_top_k=args.top_k_epochs, monitor='val_loss', mode='min')

    transform = DINOTransform()
    # we ignore object detection annotations by setting target_transform to return 0
    dataset = torchvision.datasets.VOCDetection(
        "datasets/pascal_voc",
        download=True,
        transform=transform,
        target_transform=lambda t: 0,
    )
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        devices=args.num_accelerators,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
    )
    trainer.fit(model=model, train_dataloaders=dataloader)