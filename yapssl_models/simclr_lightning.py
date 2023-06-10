import torchvision
import pytorch_lightning as pl
# from flash.core.optimizers import LARS <<< yeah the flash package has some compatibility problem

from torch import nn
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torchmetrics import MetricCollection
from timm.optim.lars import Lars
from yapssl_utils.warm_up_epochs import WarmUpLR

# Implement the warm-up epochs

class SimCLR(pl.LightningModule):
    def __init__(self, lr, warm_up_epochs, total_epochs, weight_decay):

        super().__init__()
        resnet = torchvision.models.resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(2048, 2048, 2048)
        self.lr = lr
        self.warm_up_epochs = warm_up_epochs
        self.total_epochs = total_epochs
        self.weight_decay = weight_decay

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = NTXentLoss(gather_distributed=True)

        metrics_dict = {}
        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x):

        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):

        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        return {'loss': loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):

        self.log(name='train_loss',
                 value=outputs['loss'],
                 on_step=True,
                 on_epoch=True)

        self.train_metrics.update()

        self.log_dict(self.train_metrics,
                      on_step=False,
                      on_epoch=True)

    def validation_step(self, batch, batch_idx):

        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        return {'val_loss': loss}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log(name='val_loss',
                 value=outputs['val_loss'],
                 on_step=True,
                 on_epoch=True)

        self.test_metrics.update()

        self.log_dict(self.test_metrics,
                      on_step=False,
                      on_epoch=True)

    def configure_optimizers(self):

        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        optimizer = Lars(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmUpLR(optimizer=optimizer,
                             warm_up_epochs=self.warm_up_epochs,
                             total_epochs=self.total_epochs)

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'train_loss'}

