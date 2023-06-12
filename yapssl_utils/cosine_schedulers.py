import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import AdamW

class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, last_epoch=-1):
        self.base_value = base_value
        self.final_value = final_value
        self.epochs = epochs
        self.niter_per_ep = niter_per_ep
        self.warmup_epochs = warmup_epochs
        self.start_warmup_value = start_warmup_value
        self.warmup_iters = warmup_epochs * niter_per_ep

        super(CosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            lr = np.linspace(self.start_warmup_value, self.base_value, self.warmup_iters)[self.last_epoch]
        else:
            total_iters = self.epochs * self.niter_per_ep - self.warmup_iters
            current_iter = self.last_epoch - self.warmup_iters
            lr = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + np.cos(np.pi * current_iter / total_iters))

        return [lr for _ in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()

# class AdamWCosineWD(AdamW):
#     def __init__(self, params, lr, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, last_epoch=-1, **kwargs):
#         self.base_value = base_value
#         self.final_value = final_value
#         self.epochs = epochs
#         self.niter_per_ep = niter_per_ep
#         self.warmup_epochs = warmup_epochs
#         self.start_warmup_value = start_warmup_value
#         self.warmup_iters = warmup_epochs * niter_per_ep
#         self.last_epoch = last_epoch
#         super(AdamWCosineWD, self).__init__(params, lr, **kwargs)

#     def step(self, closure=None):
#         for group in self.param_groups:
#             if self.last_epoch < self.warmup_iters:
#                 group['weight_decay'] = np.linspace(self.start_warmup_value, self.base_value, self.warmup_iters)[self.last_epoch]
#             else:
#                 total_iters = self.epochs * self.niter_per_ep - self.warmup_iters
#                 current_iter = self.last_epoch - self.warmup_iters
#                 group['weight_decay'] = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + np.cos(np.pi * current_iter / total_iters))
        
#         self.last_epoch += 1

#         super(AdamWCosineWD, self).step(closure)