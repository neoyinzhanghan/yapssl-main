import torch.nn as nn
import functools

# from ssl_models.ssl_model_mae import mae_lightning as mae
from yapssl_models import simclr_lightning as simclr
from yapssl_models import mae_lightning as mae
from yapssl_models import dino_lightning as dino

class PatchFeatureExtractor(nn.Module):
    """ Prepare a pretrained SSL model for patch feature extraction

    === Class Attributes ===
    self.chkpt: the SSL checkpoint model used for feature extraction
    self.ssl_arch: a string indicating what SSL architecture is used
    """

    def __init__(self, chkpt_fpath:str, ssl_arch:str) -> None:
        """ Initialized the PatchFeatureExtractor:
            - load and save checkpoint model as self.chkpt based on self.ssl_arch
        """
        super(PatchFeatureExtractor, self).__init__()
        self.ssl_arch = ssl_arch

        # if self.ssl_arch == 'mae':
        #     extraction_model = mae.MAELightning.load_from_checkpoint(chkpt_fpath)
        #     mae.change_mask_ratio(extraction_model, 0)
        #     extraction_model.eval()

        if self.ssl_arch == 'simclr':
            extraction_model = simclr.SimCLRLightning.load_from_checkpoint(chkpt_fpath)
            extraction_model.eval()

        if self.ssl_arch == 'mae':
            extraction_model = mae.MAELightning.load_from_checkpoint(chkpt_fpath)
            extraction_model.eval()
        if self.ssl_arch == 'dino'
            extraction_model = dino.DINOLightning.load_from_checkpoint(chkpt_fpath)
            extraction_model.eval()

        self.chkpt = extraction_model


    def forward(self, x):

        if self.ssl_arch == 'simclr':
            encoder_output = self.chkpt.backbone(x)
            batch_size = encoder_output.size(0)
            latent_feature = encoder_output.reshape(batch_size, -1) # flatten the simclr output
        
        if self.ssl_arch == 'mae':
            raise ValueError('MAE is not currently implemented for benchmarking!')
            pass # TODO 
            # encoder_output, _, _ = self.chkpt.mae.forward_encoder(x, mask_ratio=self.chkpt.mask_ratio)
            # batch_size = encoder_output.size(0)
            # latent_feature = encoder_output.reshape(batch_size, -1) # flatten the mae output

        if self.ssl_arch == 'dino':
            raise ValueError('DINO is not currently implemented for benchmarking!')
            pass # TODO 

        return latent_feature





