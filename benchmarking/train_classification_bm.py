import argparse
import os
from time import time
import matplotlib.pyplot as plt
import sys
import numpy as np

import torch
import torch.nn as nn
import timm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,\
    EarlyStopping  # DeviceStatsMonitor
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.tuner import Tuner

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC,\
    MulticlassF1Score

from wsip.LitMILClfDataModule import LitMILClfDataModule
from wsip.arch.mil_models import get_mil_model
from wsip.lit_utils import lit_clf_pred

from yaimpl.LitSupervisedModel import LitSupervisedModel
from yaimpl.LRFinderWrapper import LRFinderWrapper, SavgolSmoother
from yaimpl.utils import savefig, parse_n_cpu

# https://stackoverflow.com/questions/12151306/argparse-way-to-include-default-values-in-help
parser = argparse.\
    ArgumentParser(description='Trains an image classification model.',
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)


####################################
group = parser.add_argument_group('Dataset and paths')
####################################

group.add_argument('--label_fpath', type=str,  # required=True,
                   help='Path to csv file containing the case label csv file.')

group.add_argument('--case2wsi_fpath', type=str,  # required=True,
                   help='Path to json file mapping cases to WSI file paths.')

group.add_argument('--output_dir', type=str,  # required=True,
                   help='Directory where to save output.')


####################################
group = parser.add_argument_group('Model')
####################################

group.add_argument('--model_name', type=str, default='attn_mean_pool',
                   help="Which model to use.")


group.add_argument('--dropout', type=float, default=None,
                   help="(Optional) Dropout probability.")


group.add_argument('--encoder_n_layers', type=int, default=1,
                   help="Number of instance encoder MLP layers.")

group.add_argument('--head_n_hidden_layers', type=int, default=1,
                   help="Number of hidden layers for the head network.")

group.add_argument('--attn_latent_dim', type=int, default=1,
                   help="Latent dimension for the attention pooling network.")


parser.add_argument('--fixed_bag_size', default='max',
                    help='Fix the number of instances in each bag for training.  E.g. we randomly sample a subset of instances from each bag. This can be used to speed up the training loop. To use a batch size larger than one you must set this value. By passing in fixed_bag_size=max this will automatically set fixed_bag_size to be the largest bag size of the training set. If fixed_bag_size=q75, q90,... the fixed bag size will be the corresponding quantile of the trianing bag sizes.')

####################################
group = parser.add_argument_group('Optimizer parameters')
####################################

group.add_argument('--opt', type=str, default='sgd',
                   help="Which optimization algorithm to use.")

group.add_argument('--opt_kws', nargs='*', default={},
                   action=timm.utils.ParseKwargs)

group.add_argument("--momentum", default=0.9, type=float,
                   help="momentum")

group.add_argument("--weight_decay", default=1e-4, type=float,
                   help="weight decay")

####################################
group = parser.add_argument_group('Learning rate parameters')
####################################

group.add_argument('--lr', type=float, default=1e-3,
                   help="Learning rate.")


group.add_argument('--scale_lr_batchsize',
                   action='store_true', default=False,
                   help='Whether or not to scale the learning rate by the batch size.')

group.add_argument('--lr_finder', default='none', type=str,
                   choices=['none', 'auto', 'manual'],
                   help='How to handle automatic learning rate finder.'
                        'none means we use the specified learning rate.'
                        'auto means we run the learning rate finder and use the automatic choice.'
                        'manual means we run the automatic learning rate finder, save the plot with the suggestion then exit (i.e. do not continue on to training).')

####################################
group = parser.add_argument_group('Training details')
####################################

group.add_argument('--max_epochs', type=int, default=100,
                   help="Maximum number of epochs.")


group.add_argument('--batch_size', type=int, default=32,
                   help="Batch size.")


group.add_argument('--precision', default=32,
                   help='Precision for training. See pytorch_lightning.Trainer.')


group.add_argument('--stop_early', action='store_true', default=False,
                   help='Use early stopping to stop training.')


group.add_argument('--early_stop_patience', default=10, type=int,
                   help='Patience for early stopping')

####################################
group = parser.add_argument_group('Training logistics')
####################################

group.add_argument("--num_workers", default=1, type=float,
                   help="Number of data loading workers. Negative numbers mean TOTAL - |num_workders| + 1 e.g. -1 means use all availabel CPU cores.")

group.add_argument('--mini', action='store_true', default=False,
                   help='Run a mini experiment.')

group.add_argument('--ckpt_resume', type=str, default=None,
                   help='(Optional) Filepath to checkpoint to resume from.')

group.add_argument('--seed', type=int, default=1,
                   help='The random seed for training e.g. used to initialized network weights, order of shuffle, etc.')

group.add_argument('--num_accelerators', type=int, default=1,
                   help='Number of GPUs to use.')

group.add_argument('--accelerator', type=str, default='gpu',
                   help='Type of accelerator to use.')

group.add_argument('--lr_sched', type=str, default=None, 
                   help='Which learning rate scheduler to use.') # 'exponential'

args = parser.parse_args()



if __name__ == '__main__':

    # Set torch global precision
    torch.set_float32_matmul_precision('medium')

    profile = True

    # Number of images to sample to compute channel mean/stds
    N_IMGS_PIXEL_STATS = 1000

    # for learning rate finder
    LR_FIND_MODE = 'exponential'
    LR_FIND_N2TRY = 100

    mil_model_kws = {}

    if args.opt == 'sgd':
        opt_kws = {'nesterov': True, 'momentum': args.momentum}
    else:
        opt_kws = {}
    opt_kws.update(args.opt_kws)

    lr_sched_kws = {'expon_decay': 0.85}
    transfs_kws = {}

    loader_kws = {'num_workers': parse_n_cpu(args.num_workers)}
    # loader_kws = {}

    if args.mini:
        args.output_dir = args.output_dir + '-mini'
        N_IMGS_PIXEL_STATS = 3
        LR_FIND_N2TRY = 10

    # this avoids RuntimeError: Too many open files
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Model
    if args.model_name == 'attn_mean_pool':
        model_kws = {'dropout': args.dropout,

                    'encoder_n_layers': args.encoder_n_layers,
                    'head_n_hidden_layers': args.head_n_hidden_layers,

                    'attn_latent_dim': args.attn_latent_dim
                    }

    print('==============')
    print("Start training")
    print('==============')

    print(args)


    print("Number of available GPUs: ", torch.cuda.device_count())

    pl.seed_everything(args.seed, workers=True)


    start_time = time()


    #######################################################
    # load response data along with train/val/test splits #
    #######################################################

    datamodule = LitMILClfDataModule(label_info_fpath=args.label_fpath,
                                    case2wsi_fpath=args.case2wsi_fpath,
                                    batch_size=args.batch_size,
                                    loader_kws=loader_kws,
                                    # augment=args.augment,
                                    # augment_kws={},
                                    verbosity=0)
    datamodule.setup('fit')

    instance_feat_dim = datamodule.datasets['train'].get_feat_dim()


    ##################
    # bag size stats #
    ##################

    # Set fixed bag size
    if args.fixed_bag_size == 'max':
        fixed_bag_size = args.fixed_bag_size

    elif isinstance(args.fixed_bag_size, str)\
            and args.fixed_bag_size[0] == 'q':
        # fixed bag size is qth quntile
        # pull out quantile
        q = float(args.fixed_bag_size[1:])
        train_bag_sizes = datamodule.datasets['train'].get_bag_summary()[0]
        fixed_bag_size = int(np.percentile(train_bag_sizes, q=q))
    else:
        fixed_bag_size = int(args.fixed_bag_size)
    print("fixed bag size", fixed_bag_size)


    ########################
    # Setup loss and model #
    ########################

    # setup for class imbalance
    # sampler, loss_class_weights = get_class_imal(imbal_how=args.imbal_how,
    #                                              y=y_train_idx)

    # if args.imbal_how == 'loss_weight':
    #     aux_train_info = {'loss_class_weights': loss_class_weights}

    #     loss_class_weights = torch.from_numpy(loss_class_weights.values).float()
    #     # loss_class_weights = loss_class_weights.to(device)
    #     loss_func = nn.CrossEntropyLoss(weight=loss_class_weights)

    # else:
    #     loss_func = nn.CrossEntropyLoss()

    loss_func = nn.CrossEntropyLoss()


    #####################
    # Evaluation metrics #
    ######################

    metrics = {'accuracy': MulticlassAccuracy(num_classes=datamodule.n_classes),
            'auc': MulticlassAUROC(num_classes=datamodule.n_classes),
            'f1': MulticlassF1Score(num_classes=datamodule.n_classes)}

    ###############
    # Setup model #
    ###############

    model = get_mil_model(instance_dim=instance_feat_dim,
                        n_out=datamodule.n_classes,
                        model_name=args.model_name,
                        model_kws=mil_model_kws)


    #########################
    # Setup lightning model #
    #########################

    lit_model = LitSupervisedModel(model=model,
                                loss_func=loss_func,
                                metrics=metrics,
                                opt=args.opt,
                                opt_kws=opt_kws,
                                lr=args.lr,
                                lr_sched=args.lr_sched,
                                lr_sched_kws=lr_sched_kws,
                                weight_decay=args.weight_decay,
                                batch_size=args.batch_size,
                                scale_lr_batchsize=args.scale_lr_batchsize
                                )

    #################
    # Setup trainer #
    #################


    ckpt = ModelCheckpoint(save_weights_only=False,
                        save_last=False,
                        save_on_train_epoch_end=False,
                        verbose=True,
                        mode="min",
                        monitor="val_loss")
    callbacks = [LearningRateMonitor(logging_interval="step"),
                ckpt,
                ]

    # maye add early stopping
    if args.stop_early:
        callbacks.append(
                EarlyStopping(monitor='val_loss', mode='min',
                            patience=args.early_stop_patience)
                )


    if profile:
        # callbacks.extend([DeviceStatsMonitor(cpu_stats=True)])
        profiler = SimpleProfiler(filename='prifiler.txt')
    else:
        profiler = None

    if args.mini:
        mini_kws = {'limit_train_batches': 3,
                    'limit_val_batches': 3,
                    'limit_test_batches': 3,
                    'limit_predict_batches': 3}
        args.max_epochs = 3
    else:
        mini_kws = {}

    trainer = pl.Trainer(accelerator=args.accelerator,
                        devices=args.num_accelerators,
                        strategy='auto',
                        logger=True,
                        profiler=profiler,
                        enable_checkpointing=True,
                        callbacks=callbacks,
                        default_root_dir=args.output_dir,
                        check_val_every_n_epoch=1,

                        max_epochs=args.max_epochs,
                        **mini_kws,

                        accumulate_grad_batches=1)


    ###########################
    # Find optimal batch size #
    ###########################


    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.tuner.tuning.Tuner.html
    # TODO: need to give lit_model a datamodel or a batch_size argument
    # trainer.tuner.scale_batch_size(model=lit_model,
    #                                train_dataloaders=train_loader,
    #                                mode='power',
    #                                steps_per_trial=3,
    #                                init_val=2,
    #                                max_trials=25,
    #                                batch_arg_name='batch_size')


    ##############################
    # Find optimal learning rate #
    ##############################

    if args.lr_finder.lower() in ['auto', 'manual']:

        tuner = Tuner(trainer)

        pt_lr_finder = tuner.lr_find(model=lit_model,
                                    datamodule=datamodule,

                                    mode=LR_FIND_MODE,
                                    num_training=LR_FIND_N2TRY,
                                    min_lr=1e-8,
                                    max_lr=1,
                                    early_stop_threshold=4.0,
                                    update_attr=False)

        lr_finder = LRFinderWrapper(lr_finder=pt_lr_finder,
                                    smoother=SavgolSmoother(window_length=3,
                                                            polyorder=1))

        suggested_lr = lr_finder.get_lr_suggestion()
        print("Suggested learning rate: {}".format(suggested_lr))

        # save plot
        plt.figure(figsize=(5, 5))
        lr_finder.plot(suggest=True)
        savefig(os.path.join(args.output_dir, 'learning_rate_finder.png'))

        if args.lr_finder.lower() == 'manual':
            # exit so user can manually input the learning rate
            sys.exit(0)
        if args.lr_finder.lower() == 'auto':
            lit_model.lr = suggested_lr

    train_dataset = datamodule.datasets['train']
    for idx in range(len(train_dataset)):
        train_dataset[idx]

    if __name__ == '__main__':
        ##############
        # Fit model! #
        ##############
        trainer.fit(model=lit_model,
                    datamodule=datamodule,
                    ckpt_path=args.ckpt_resume)

        print("Best checkpoint {}, with val loss = {}".format(ckpt.best_model_path,
                                                            ckpt.best_model_score))

        ###########
        # Predict #
        ###########

        # for predictions DO NOT use distributed data parallel
        pred_trainer = pl.Trainer(accelerator=args.accelerator,
                                devices=1,
                                default_root_dir=args.output_dir,
                                **mini_kws)

        lit_clf_pred(model=lit_model, trainer=pred_trainer, datamodule=datamodule,
                    ckpt_path=ckpt.best_model_path,
                    save=True, stub='best')

        lit_clf_pred(model=lit_model, trainer=pred_trainer, datamodule=datamodule,
                    # ckpt_path='last',
                    save=True, stub='last')
