import torch.optim.lr_scheduler as lr_scheduler

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