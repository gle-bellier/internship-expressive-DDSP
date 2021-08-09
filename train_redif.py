from redif.model import Model
from redif.dataset import Dataset
from redif.ema import EMAModelCheckPoint
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from effortless_config import Config


class args(Config):
    CKPT = None


args.parse_args()

dataset = Dataset(256)
val_n = len(dataset) // 50
train, val = random_split(dataset, [len(dataset) - val_n, val_n])

if args.CKPT is not None:
    model = Model.load_from_checkpoint(args.CKPT, strict=False)
else:
    model = Model(2, 128, [128, 192, 256, 384, 512])

model.set_noise_schedule()
model.transform.compute_stats(dataset.e_f0, dataset.e_lo)
model.ddsp = torch.jit.load("ddsp_violin_pretrained.ts")

trainer = pl.Trainer(gpus=1,
                     check_val_every_n_epoch=10,
                     callbacks=[EMAModelCheckPoint(
                         model,
                         filename="ema",
                     )],
                     max_epochs=100000,
                     resume_from_checkpoint=args.CKPT)
trainer.fit(
    model,
    DataLoader(train, 16, True, drop_last=True),
    DataLoader(val, 16, False),
)
