import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class EMAModelCheckPoint(ModelCheckpoint):
    """
    PytorchLightning callback inherited from ModelCheckpoint
    Tracks an EMA version of the weight of a model.
    Substitute the weights with its shadow during the validation pass.
    """
    def __init__(self, model: torch.nn.Module, alpha=.999, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.model = model
        self.alpha = alpha

    def on_train_batch_end(self, *args, **kwargs):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    self.shadow[n] *= self.alpha
                    self.shadow[n] += (1 - self.alpha) * p.data

    def on_validation_epoch_start(self, *args, **kwargs):
        self.swap()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.swap()

    def swap(self):
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                tmp = p.data.clone()
                p.data.copy_(self.shadow[n])
                self.shadow[n] = tmp

    def save_checkpoint(self, *args, **kwargs):
        self.swap()
        super().save_checkpoint(*args, **kwargs)
        self.swap()
