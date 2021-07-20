import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from torch import nn
from utils import Identity, ConvBlock
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from unet_dataset import UNet_Dataset
import matplotlib.pyplot as plt
import os, sys


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.mp = nn.MaxPool1d(kernel_size=2)
        pass

    def forward(self, x):
        x = self.conv1(x)
        ctx = torch.clone(x)
        x = self.conv2(x)
        out = self.mp(x)
        return out, ctx


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.gru = nn.GRU(input_size=out_channels,
                          hidden_size=out_channels,
                          batch_first=True)

    def forward(self, x):

        x = self.conv1(x)

        # permuting for GRU : B,C,T -> B, T, C

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)

        out = self.conv2(x)
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(2 * out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=1,
                               kernel_size=3,
                               padding=1))
        self.conv_ctx = nn.ConvTranspose1d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           stride=1,
                                           kernel_size=3,
                                           padding=1)

        self.lr = nn.LeakyReLU()

    def add_ctx(self, x, ctx):
        # # crop context (y)
        # d_shape = (ctx.shape[-1] - x.shape[-1]) // 2
        # crop = ctx[:, :, d_shape:d_shape + x.shape[2]]
        # #concatenate
        out = torch.cat([x, ctx], 1)
        return out

    def forward(self, x, ctx):
        x = self.up_conv(x)
        ctx = self.conv_ctx(ctx)
        x = self.add_ctx(x, ctx)
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class UNet_RNN(pl.LightningModule):
    def __init__(self, channels, scalers, None):
        super().__init__()
        self.save_hyperparameters()

        down_channels = channels
        up_channels = channels[::-1]

        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]

        self.scalers = scalers
        self.ddsp = None
        self.val_idx = 0

        self.down_blocks = nn.ModuleList([
            DBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_in,
                                                 self.down_channels_out)
        ])

        self.bottleneck = Bottleneck(in_channels=self.down_channels_out[-1],
                                     out_channels=self.up_channels_in[0])

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.up_channels_in,
                                                 self.up_channels_out)
        ])

    def down_sampling(self, x):
        l_ctx = []
        for i in range(len(self.down_blocks)):
            x, ctx = self.down_blocks[i](x)
            l_ctx = [ctx] + l_ctx

        return x, l_ctx

    def up_sampling(self, x, l_ctx):

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, l_ctx[i])
        return x

    def forward(self, x):
        # permute from B, T, C -> B, C, T
        x = x.permute(0, 2, 1)
        out, l_ctx = self.down_sampling(x)
        out = self.bottleneck(out)
        out = self.up_sampling(out, l_ctx)
        # permute from B, C, T -> B, T, C
        out = out.permute(0, 2, 1)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch, batch_idx):
        model_input, target = batch
        loss = self.compute_loss(model_input, target)
        self.log("loss", loss)
        return loss

    def compute_loss(self, model_input, target):
        out = self.forward(model_input)

        pred_f0, pred_lo = torch.split(out, 1, -1)
        target_f0, target_lo = torch.split(target, 1, -1)

        loss_f0 = nn.functional.mse_loss(pred_f0, target_f0)
        loss_lo = nn.functional.mse_loss(pred_lo, target_lo)

        return loss_f0 + loss_lo

    def validation_step(self, batch, batch_idx):

        model_input, target = batch
        loss = self.compute_loss(model_input, target)
        self.log("val_loss", loss)

        return (model_input, target)

    def post_process(self, out):

        f0, l0 = torch.split(out, 1, -1)

        f0 = f0.reshape(-1, 1).cpu().numpy()
        l0 = l0.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        l0 = self.scalers[1].inverse_transform(l0).reshape(-1)

        return f0, l0

    def validation_epoch_end(self, inputs):

        model_input, target = inputs[-1]  # first elt of last batch
        model_input = model_input[0:1]
        target = target[0:1]
        self.val_idx += 1

        if self.val_idx % 20:
            return

        device = next(iter(self.parameters())).device
        out = self.neural_pass(model_input)

        f0, lo = self.post_process(out)
        midi_f0, midi_lo = self.post_process(model_input)
        target_f0, target_lo = self.post_process(target)

        plt.plot(f0)
        plt.plot(midi_f0)
        plt.plot(target_f0)
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo)
        plt.plot(midi_lo)
        plt.plot(target_lo)
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        if self.ddsp is not None:
            f0 = torch.from_numpy(f0).float().reshape(1, -1, 1).to("cuda")
            lo = torch.from_numpy(lo).float().reshape(1, -1, 1).to("cuda")
            signal = self.ddsp(f0, lo)
            signal = signal.reshape(-1).cpu().numpy()

            self.logger.experiment.add_audio(
                "generation",
                signal,
                self.val_idx,
                16000,
            )


if __name__ == "__main__":

    list_transforms = [
        (MinMaxScaler, ),
        (QuantileTransformer, 30),
    ]

    dataset = UNet_Dataset(list_transforms=list_transforms, n_sample=2048)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 16, 512, 1024]

    model = UNet_RNN(scalers=dataset.scalers, channels=down_channels)

    model.ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()
    tb_logger = pl_loggers.TensorBoardLogger('logs/unet-rnn/')

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=10000,
        logger=tb_logger)

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )
