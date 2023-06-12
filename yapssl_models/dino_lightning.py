# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import pytorch_lightning as pl
import torch

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from torchmetrics import MetricCollection

from yapssl_utils.cosine_schedulers import CosineScheduler
from torch.optim import AdamW

class DINO(pl.LightningModule):
    def __init__(self,
                 lr,
                 min_lr,
                 epochs,
                 niter_per_ep,
                 lr_warm_up_epochs = 10,
                 sub_patch_size = 8,
                 temp_student = 0.1,
                 temp_teacher_start = 0.04, # cosine schedule from 0.04 to 0.07 over the first 30 epochs
                 temp_teacher_end = 0.07,
                 temp_teacher_warm_up_epochs =30,
                 weight_decay=(0.04 + 0.4)/2):
        
        super().__init__()

        # this is if you want to use resnet18 backbone
        # resnet = torchvision.models.resnet18()
        # backbone = nn.Sequential(*list(resnet.children())[:-1])
        # input_dim = 512

        # instead of a resnet you can also use a vision transformer backbone as in the
        # original paper (you might have to reduce the batch size in this case):
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        input_dim = backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.lr = lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.niter_per_ep = niter_per_ep
        self.lr_warm_up_epochs = lr_warm_up_epochs
        self.sub_patch_size = sub_patch_size
        self.temp_student = temp_student
        self.temp_teacher_start = temp_teacher_start
        self.temp_teacher_end = temp_teacher_end
        self.temp_teacher_warm_up_epochs = temp_teacher_warm_up_epochs
        self.weight_decay = weight_decay

        self.criterion = DINOLoss(output_dim=2048, 
                                  warmup_teacher_temp_epochs=self.temp_teacher_warm_up_epochs,
                                  warmup_teacher_temp = self.temp_teacher_start,
                                  teacher_temp = self.temp_teacher_end,
                                  student_temp = self.temp_student)

        metrics_dict = {}
        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        return {'loss': loss}

    def on_training_batch_end(self, outputs, batch, batch_idx, dataloader_idx):

        self.log(name='train_loss',
                 value=outputs['loss'],
                 on_step=True,
                 on_epoch=True)

        self.train_metrics.update()

        self.log_dict(self.train_metrics,
                      on_step=False,
                      on_epoch=True)


    def validation_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)

        return {'val_loss': loss}


    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.log(name='val_loss',
                 value=outputs['val_loss'],
                 on_step=True,
                 on_epoch=True)

        self.train_metrics.update()

        self.log_dict(self.train_metrics,
                      on_step=False,
                      on_epoch=True)
        
    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.lr,
                          weight_decay=self.weight_decay)

        scheduler = CosineScheduler(optimizer, 
                                    base_value=self.lr, 
                                    final_value=self.min_lr, 
                                    epochs=self.epochs,
                                    niter_per_ep=self.niter_per_ep, 
                                    warmup_epochs=self.lr_warm_up_epochs)
        
        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'}