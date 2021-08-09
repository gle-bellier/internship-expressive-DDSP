from redif.model import Model
from redif.dataset import Dataset
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

dataset = Dataset(2048)
val_n = len(dataset) // 50
train, val = random_split(dataset, [len(dataset) - val_n, val_n])

model = Model(2, 128, [128, 256, 384, 512])
model.set_noise_schedule()
model.transform.compute_stats(dataset.e_f0, dataset.e_lo)
model.ddsp = torch.jit.load("ddsp_violin_pretrained.ts")

trainer = pl.Trainer(
    gpus=1,
    check_val_every_n_epoch=10,
)
trainer.fit(
    model,
    DataLoader(train, 16, True, drop_last=True),
    DataLoader(val, 16, False),
)
