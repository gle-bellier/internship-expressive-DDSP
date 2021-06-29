import torch
import pytorch_lightning as pl
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
        print("x shape {}".format(x.shape))
        x = self.conv1(x)
        x = self.conv2(x)

        ctx = torch.clone(x)
        out = self.mp(x)

        print("out {} ! ctx {}".format(out.shape, ctx.shape))
        return out, ctx


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, n_sample):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.gru = nn.GRU(input_size=n_sample,
                          hidden_size=n_sample,
                          batch_first=True)

    def forward(self, x):
        print("Bottleneck")
        print("x shape {}".format(x.shape))
        x = self.conv1(x)

        # permuting for GRU : B,C,T -> B, T, C
        x = x.permute(0, 2, 1)
        x = self.gru(x)
        x = x.permute(0, 2, 1)

        out = self.conv2(x)
        print("out shape {}".format(out.shape))
        return out


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               stride=1,
                               kernel_size=3,
                               padding=1))

        self.lr = nn.LeakyReLU()

    def add_ctx(self, x, ctx):
        # crop context (y)
        d_shape = (ctx.shape[-1] - x.shape[-1]) // 2
        crop = ctx[:, :, d_shape:d_shape + x.shape[2]]
        #concatenate
        out = torch.cat([x, crop], 1)
        return out

    def forward(self, x, ctx):
        print("x shape {} ! ctx {}".format(x.shape, ctx.shape))
        x = self.up_conv(x)
        print("x shape {} ! ctx {}".format(x.shape, ctx.shape))
        x = self.add_ctx(x, ctx)
        print("x shape {}".format(x.shape))
        x = self.conv1(x)
        out = self.conv2(x)
        print("out shape {}".format(out.shape))
        return out


class UNet_RNN(pl.LightningModule):
    def __init__(self, channels, n_sample, scalers, ddsp):
        super().__init__()
        #self.save_hyperparameters()

        down_channels = channels
        up_channels = channels[::-1]

        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]

        self.scalers = scalers
        self.ddsp = ddsp
        self.val_idx = 0

        self.down_blocks = nn.ModuleList([
            DBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.down_channels_in,
                                                 self.down_channels_out)
        ])

        self.bottleneck = Bottleneck(in_channels=self.down_channels_out[-1],
                                     out_channels=self.up_channels_in[0],
                                     n_sample=n_sample)

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.up_channels_in,
                                                 self.up_channels_out)
        ])

    def down_sampling(self, x):
        l_out = []
        l_ctx = []
        for i in range(len(self.down_blocks)):
            x, ctx = self.down_blocks[i](x)
            l_ctx = [ctx] + l_ctx

        return x, l_ctx

    def up_sampling(self, x, l_ctx):

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, l_ctx[i])
        return x

    def neural_pass(self, x, noise_level):

        # permute from B, T, C -> B, C, T
        x = x.permute(0, 2, 1)

        out, l_ctx = self.down_sampling(x)

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

    def validation_step(self, batch, batch_idx):

        model_input, target = batch
        loss = self.compute_loss(model_input, target)
        self.log("val_loss", loss)

        # returns cdt for validation end epoch
        return model_input

    def post_process(self, out):

        f0, l0 = torch.split(out, 1, 1)

        f0 = f0.reshape(-1).cpu().numpy()
        l0 = l0.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        l0 = self.scalers[1].inverse_transform(l0).reshape(-1)

        return f0, l0

    def validation_epoch_end(self, inputs):

        model_in = inputs[-1][0]  # first elt of last batch
        self.val_idx += 1

        if self.val_idx % 100:
            return

        device = next(iter(self.parameters())).device
        out = self.neural_pass(model_in)
        f0, lo = self.post_process(out)

        plt.plot(f0)
        self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)
        plt.plot(lo)
        self.logger.experiment.add_figure("loudness", plt.gcf(), self.val_idx)

        if self.ddsp is not None:
            f0 = torch.from_numpy(f0).float().reshape(1, -1, 1)
            lo = torch.from_numpy(lo).float().reshape(1, -1, 1)
            signal = self.ddsp(f0, lo)
            signal = signal.reshape(-1).numpy()

            self.logger.experiment.add_audio(
                "generation",
                signal,
                self.val_idx,
                16000,
            )


if __name__ == "__main__":

    ddsp = torch.jit.load("../ddsp_debug_pretrained.ts").eval()

    # trainer = pl.Trainer(
    #     gpus=1,
    #     callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
    #     max_epochs=10000,
    # )
    # list_transforms = [
    #     (MinMaxScaler, ),
    #     (QuantileTransformer, 30),
    # ]

    # len_sample = 2048
    # dataset = Dataset(list_transforms=list_transforms, n_sample=2048)
    # val_len = len(dataset) // 20
    # train_len = len(dataset) - val_len

    # train, val = random_split(dataset, [train_len, val_len])

    # down_channels = [2, 16, 64, 256]
    # ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

    # model = UNet_RNN(scalers=dataset.scalers,
    #                  channels=down_channels,
    #                  ddsp=ddsp)

    # model.set_noise_schedule()

    # trainer.fit(
    #     model,
    #     DataLoader(train, 32, True),
    #     DataLoader(val, 32),
    # )
