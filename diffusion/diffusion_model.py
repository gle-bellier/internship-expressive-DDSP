import torch
import pytorch_lightning as pl
from torch import nn


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.up_sampling = nn.Upsample(scale_factor=2)

        self.blck = nn.Sequential(
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=2), nn.LeakyReLU())

    def forward(self, x):
        print(x.shape)
        x = self.up_sampling(x)
        print(x.shape)
        x = self.conv1d(x)
        print(x.shape)
        x = self.blck(x)
        print(x.shape)
        x = self.blck(x)
        print(x.shape)

        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.conv1d = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3)
        self.mp = nn.MaxPool1d(kernel_size=2)
        pass

    def forward(self, x):
        print(x.shape)
        x = self.conv1d(x)
        print(x.shape)
        x = self.lr(x)
        print(x.shape)
        x = self.mp(x)
        print(x.shape)

        return x


class UNET(pl.LightningModule):
    def __init__(self, in_size, hidden_size, out_size, scalers):
        super().__init__()
        self.save_hyperparameters()
        self.scalers = scalers

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
            weight_decay=.01,
        )

    def forward(self, x):
        x = torch.permute(0, 2, 1)
        return x

    def get_losses(self, pred_f0, pred_cents, pred_loudness, target_f0,
                   target_cents, target_loudness):
        pass

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input.float())

        pred_f0, pred_cents, pred_loudness = self.split_predictions(prediction)
        target_f0, target_cents, target_loudness = torch.split(target, 1, -1)

        loss_f0, loss_cents, loss_loudness = self.get_losses(
            pred_f0,
            pred_cents,
            pred_loudness,
            target_f0,
            target_cents,
            target_loudness,
        )

        self.log("loss_f0", loss_f0)
        self.log("loss_cents", loss_cents)
        self.log("loss_loudness", loss_loudness)

        return loss_f0 + loss_cents + loss_loudness

    @torch.no_grad()
    def generation_loop(self, x, infer_pitch=False):
        pass

    def apply_inverse_transform(self, x, idx):
        scaler = self.scalers[idx]
        x = x.cpu()
        out = scaler.inverse_transform(x.reshape(-1, 1))
        out = torch.from_numpy(out).to("cuda")
        out = out.unsqueeze(0)
        return out.float()

    def get_audio(self, model_input, target):
        pass

    def validation_step(self, batch, batch_idx):
        model_input, target = batch
        prediction = self.forward(model_input.float())
