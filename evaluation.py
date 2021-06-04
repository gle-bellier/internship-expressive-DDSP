import torch
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, model, sr=100):
        self.model = model
        self.sr = sr

    def evaluate(self,
                 model_input,
                 target,
                 PLOT=False,
                 SCORE=True,
                 reduction="mean"):

        out = self.model.predic(model_input)

        if PLOT:
            self.plot(out, target)

        if SCORE:
            self.score(out, target, reduction)

    def plot(self, out, target):

        t = torch.arange(0, out.size(1), 1) / self.sr

        out_f0, out_loudness = out.split(1, -1)
        target_f0, target_loudness = target.split(1, -1)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Model predictions")

        ax1.plot(t, out_f0.squeeze(), label="Model")
        ax1.plot(t, target_f0.squeeze(), label="Target")
        ax1.set_title("Frequency")

        ax2.plot(t, out_loudness.squeeze(), label="Model")
        ax2.plot(t, target_loudness.squeeze(), label="Target")
        ax2.set_title("Loudness")

        plt.show()

    def score(self, out, target, reduction):

        out_f0, out_loudness = out.split(1, -1)
        target_f0, target_loudness = target.split(1, -1)

        d_cents = 1200 * torch.log2(out_f0.squeeze() / target_f0.squeeze())

        if reduction == "mean":
            return torch.mean(d_cents)
        elif reduction == "median":
            return torch.media(d_cents)
        elif reduction == "sum":
            return torch.sum(d_cents)
        else:
            print("ERROR reduction type")
            return None
