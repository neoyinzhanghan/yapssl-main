import torch
import pytorch_lightning as pl
import timm.optim.optim_factory as optim_factory
import torch.optim.lr_scheduler as lr_scheduler

from torch import optim, nn, utils, Tensor
from torchmetrics import MetricCollection
from ssl_models.ssl_model_mae.mae_models import MAE

# Implement the warm-up epochs
class WarmUpLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up_epochs, total_epochs, last_epoch=-1):
        self.warm_up_epochs = warm_up_epochs
        self.total_epochs = total_epochs
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warm_up_epochs:
            warm_up_factor = (self.last_epoch + 1) / self.warm_up_epochs
            return [base_lr * warm_up_factor for base_lr in self.base_lrs]
        else:
            remaining_epochs = self.total_epochs - self.warm_up_epochs
            decay_factor = (self.last_epoch - self.warm_up_epochs + 1) / remaining_epochs
            return [base_lr * (1 - decay_factor) for base_lr in self.base_lrs]


# define the LightningModule
class MAELightning(pl.LightningModule):
    def __init__(self, lr=0.001,
                 warm_up_epochs=40,
                 total_epochs=800,
                 mask_ratio=0.75,
                 weight_decay=0.05):
        super().__init__()
        model = MAE()
        self.mae = model
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.weight_decay = weight_decay
        self.warm_up_epochs = warm_up_epochs
        self.total_epochs = total_epochs

        metrics_dict = {}

        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        x, y = batch

        latent, mask, ids_restore = self.mae.forward_encoder(x, mask_ratio=self.mask_ratio)
        pred = self.mae.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        loss = self.mae.forward_loss(x, pred, mask)

        return {'loss': loss,
                'pred': pred,
                'target': x}

    def on_train_batch_end(self, outputs, batch, batch_idx):

        self.log(name='train_loss',
                 value=outputs['loss'],
                 on_step=True,
                 on_epoch=True)

        self.train_metrics.update(preds=outputs['pred'],
                                  target=outputs['target'])

        self.log_dict(self.train_metrics,
                      on_step=False,
                      on_epoch=True)

    def validation_step(self, batch, batch_idx):
        # this test_step defines the test loop

        x, y = batch
        with torch.cuda.amp.autocast():
            loss_out, pred, mask = self.mae(x, mask_ratio=self.mask_ratio)

        target = self.mae.patchify(x)

        loss = (pred - target) ** 2 # this is essentially the pixel-wise L2 reconstruction loss
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        val_loss_value_all = loss.sum()/loss.numel()
        val_loss_value_non_masked = (loss * (1 - mask)).sum()/(loss.numel() - mask.sum())
        val_loss_value_masked = (loss * mask).sum()/(mask.sum())

        return {'val_loss_masked': val_loss_value_masked,
                'val_loss_all': val_loss_value_all,
                'val_loss_non_masked': val_loss_value_non_masked,
                'pred': pred,
                'target': target}

    def on_validation_batch_end(self, outputs, batch, batch_idx):

        self.log(name='val_loss_masked',
                 value=outputs['val_loss_masked'],
                 on_step=False,
                 on_epoch=True)

        self.log(name='val_loss_non_masked',
                 value=outputs['val_loss_non_masked'],
                 on_step=False,
                 on_epoch=True)

        self.log(name='val_loss_all',
                 value=outputs['val_loss_all'],
                 on_step=False,
                 on_epoch=True)

        self.val_metrics.update(preds=outputs['pred'],
                                target=outputs['target'])

        self.log_dict(self.val_metrics,
                      on_step=False,
                      on_epoch=True)

    def configure_optimizers(self):
        param_groups = optim_factory.add_weight_decay(self.mae, self.weight_decay)
        optimizer = optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        scheduler = WarmUpLR(optimizer=optimizer,
                             warm_up_epochs=self.warm_up_epochs,
                             total_epochs=self.total_epochs)

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss_masked'}

def change_mask_ratio(lightning_mae_model:MAELightning, new_mask_ratio:float) -> None:
    """ Modify the mask_ratio attribute of the lightning_mae_model to new_mask_ratio

    Preconditions:
    - 0<= new_mask_ratio <= 1

    Raise ValueError if preconditions fail
    """

    if not (0<= new_mask_ratio <= 1):
        raise ValueError("mask_ratio must be between 0 and 1")

    else:
        lightning_mae_model.mask_ratio = new_mask_ratio