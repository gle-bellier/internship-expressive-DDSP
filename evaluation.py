import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


class Evaluator:
    def __init__(self, sr=100):
        self.sr = sr

    def evaluate(self,
                 out_f0,
                 out_loudness,
                 target_f0,
                 target_loudness,
                 PLOT=False,
                 SCORE=True,
                 reduction="mean"):

        if PLOT:
            self.plot(out_f0, out_loudness, target_f0, target_loudness)

        if SCORE:
            return self.score(out_f0, out_loudness, target_f0, target_loudness,
                              reduction)

    def plot(self, out_f0, out_loudness, target_f0, target_loudness):

        t = torch.arange(0, out_f0.size(1), 1) / self.sr

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Model predictions")

        ax1.plot(t, out_f0.squeeze(), label="Model")
        ax1.plot(t, target_f0.squeeze(), label="Target")
        ax1.set_title("Frequency")
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax2.plot(t, out_loudness.squeeze(), label="Model")
        ax2.plot(t, target_loudness.squeeze(), label="Target")
        ax2.set_title("Loudness")
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def score(self, out_f0, out_loudness, target_f0, target_loudness,
              reduction):

        d_cents = 1200 * torch.log2(
            torch.abs(out_f0.squeeze()) / target_f0.squeeze()[1:])

        d_cents = torch.abs(d_cents)
        if reduction == "mean":
            return torch.mean(d_cents).numpy()
        elif reduction == "median":
            return torch.media(d_cents).numpy()
        elif reduction == "sum":
            return torch.sum(d_cents).numpy()
        else:
            print("ERROR reduction type")
            return None

    def listen(self,
               out_f0,
               out_loudness,
               target_f0,
               target_loudness,
               ddsp,
               saving_path=None,
               resynth=False):

        model_audio = ddsp(out_f0, out_loudness)
        target_audio = ddsp(target_f0, target_loudness)

        if saving_path is not None:
            write(saving_path, 16000, model_audio.reshape(-1).numpy())

        if resynth is not None:
            filename = saving_path[:-4] + "-resynth.wav"
            write(filename, 16000, target_audio.reshape(-1).numpy())

        return model_audio
