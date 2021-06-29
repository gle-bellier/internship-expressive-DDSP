import torch
import pytorch_lightning as pl
from torch import nn
from utils import FiLM, Identity
from diffusion import DiffusionModel
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from diffusion_dataset import DiffusionDataset
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.lr = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        out = self.lr(x)
        return out


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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.gru = nn.GRU(input_size=512, hidden_size=203, batch_first=True)

    def forward(self, x):
        print("Bottleneck")
        print("x shape {}".format(x.shape))
        x = self.conv1(x)
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
    def __init__(self, down_channels, up_channels, scalers, ddsp):
        super().__init__()
        #self.save_hyperparameters()
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

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=channels_in, out_channels=channels_out)
            for channels_in, channels_out in zip(self.up_channels_in,
                                                 self.up_channels_out)
        ])

    def down_sampling(self, list_blocks, x):
        l_out = []
        for i in range(len(list_blocks)):
            x = list_blocks[i](x)
            l_out = l_out + [x]
        return l_out

    def up_sampling(self, x, l_film_pitch, l_film_noisy):
        l_film_pitch = l_film_pitch[::-1]
        l_film_noisy = l_film_noisy[::-1]

        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, l_film_pitch[i], l_film_noisy[i])
        return x

    def neural_pass(self, x, cdt, noise_level):

        # permute from B, T, C -> B, C, T
        noisy = x.permute(0, 2, 1)
        pitch = cdt.permute(0, 2, 1)

        l_out_pitch = self.down_sampling(self.down_blocks_pitch, pitch)
        l_out_noisy = self.down_sampling(self.down_blocks_noisy, noisy)

        l_film_pitch = self.film(self.films_pitch, l_out_pitch, None)
        l_film_noisy = self.film(self.films_noisy, l_out_noisy, noise_level)

        hiddens = self.cat_hiddens(l_out_pitch[-1], l_out_noisy[-1])
        out = self.up_sampling(hiddens, l_film_pitch, l_film_noisy)

        # permute from B, C, T -> B, T, C
        out = out.permute(0, 2, 1)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4)

    def training_step(self, batch, batch_idx):
        model_input, cdt = batch
        loss = self.compute_loss(model_input, cdt)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        # loss = self.compute_loss(batch, batch_idx) Why ??

        model_input, cdt = batch
        loss = self.compute_loss(model_input, cdt)
        self.log("val_loss", loss)

        # returns cdt for validation end epoch
        return cdt

    def post_process(self, out):

        # change range [-1, 1] -> [0, 1]
        out = out / 2 + .5

        f0, l0 = torch.split(out, 1, 1)

        f0 = f0.reshape(-1).cpu().numpy()
        l0 = l0.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        l0 = self.scalers[1].inverse_transform(l0).reshape(-1)

        return f0, l0

    def validation_epoch_end(self, cdt):
        self.val_idx += 1

        if self.val_idx % 100:
            return

        device = next(iter(self.parameters())).device
        x = torch.zeros_like(cdt).to(device)
        x = self.sample(x, cdt)
        f0, lo = self.post_process(x)

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

    @torch.no_grad()
    def sample(self, x, cdt):
        x = torch.randn_like(x)
        for i in range(self.n_step)[::-1]:
            x = self.inverse_dynamics(x, cdt, i)
        return x


if __name__ == "__main__":

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        max_epochs=10000,
    )
    list_transforms = [
        (MinMaxScaler, ),
        (QuantileTransformer, 30),
    ]

    len_sample = 2048
    dataset = DiffusionDataset(list_transforms=list_transforms)
    val_len = len(dataset) // 20
    train_len = len(dataset) - val_len

    train, val = random_split(dataset, [train_len, val_len])

    down_channels = [2, 16, 64, 256]
    up_channels = [256, 64, 16, 2]
    ddsp = torch.jit.load("ddsp_debug_pretrained.ts").eval()

    model = UNet_RNN(scalers=dataset.scalers,
                     down_channels=down_channels,
                     up_channels=up_channels,
                     ddsp=ddsp)

    model.set_noise_schedule()

    trainer.fit(
        model,
        DataLoader(train, 32, True),
        DataLoader(val, 32),
    )
