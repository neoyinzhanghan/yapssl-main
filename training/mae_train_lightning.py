import os
import argparse

import pytorch_lightning as pl
from lightly.data.multi_view_collate import MultiViewCollate
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset
from yapssl_models.mae_lightning import MAE
# from ssl_models.utils.grab_pt_patches import MySSLDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from yaimpl.utils import parse_n_cpu
from lightly.transforms.mae_transform import MAETransform



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

parser.add_argument('--batch_size', default=64, type=int, #4096 is the paper default eff_batch_size
                    help='Batch size for training')
parser.add_argument('--test_batch_size', default=64, type=int,
                    help='Batch size for testing')
parser.add_argument('--epochs', default=800, type=int,
                    help='total number of epochs during training')
parser.add_argument('--blr', default=1.5e-4, type=float,
                    help='this base learning rate is used to compute the actual learning rate')
parser.add_argument('--max_num_batches', default=None, type=int,
                    help='the maximum number of batches per epoch')
parser.add_argument('--weight_decay', default=0.05, type=float,
                    help='In deep learning, weight decay is a regularization technique used to prevent overfitting by adding a penalty to the loss function, which effectively reduces the magnitude of the weights in the model.')
parser.add_argument('--warm_up_epochs', default=40, type=int,
                    help='the number of epochs for a linear training warm-up')
parser.add_argument('--accum_grad', default=1, type=int,
                    help='accumulate gradients for accum_grad batches')

####################################
group = parser.add_argument_group('Hardwares Configuration')
####################################

parser.add_argument('--num_workers', default=1, type=int,
                    help='the number of workers in parsing batches')
group.add_argument('--num_accelerators', default=1, type=int,
                    help='the number of accelerators in distributed training')
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

    eff_batch_size = args.batch_size * args.num_accelerators * args.accum_grad
    lr = args.blr * eff_batch_size/256

    # DATA AUGMENTATION

    transform = MAETransform()
    dataset_train = LightlyDataset(os.path.join(os.path.expanduser(args.data_path),'train'), transform=transform)
    dataset_test = LightlyDataset(os.path.join(os.path.expanduser(args.data_path),'test'), transform=transform)

    # DATALOADERS

    collate_fn = MultiViewCollate()    
    train_loader = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True,
                              drop_last=True,
                              num_workers=parse_n_cpu(args.num_workers))

    test_loader = DataLoader(dataset_test,
                             batch_size=args.test_batch_size,
                             collate_fn=collate_fn,
                             shuffle=False,
                             drop_last=True,
                             num_workers=parse_n_cpu(args.num_workers))
    
    # MODEL
    model = MAE(lr=lr, warm_up_epochs=args.warm_up_epochs, total_epochs=args.epochs, weight_decay=args.weight_decay)

    # CHECKPOINT CALLBACK SETTING
    checkpoint_callback = ModelCheckpoint(save_top_k=args.top_k_epochs, monitor='val_loss', mode='min')


    # Train with DDP on multiple gpus. Distributed sampling is also enabled with
    # replace_sampler_ddp=True.
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        devices=args.num_accelerators,
        accelerator=args.type_accelerator,
        accumulate_grad_batches=args.accum_grad
        )

    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=test_loader,
                )

