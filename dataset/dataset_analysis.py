import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


class Analyzer:
    def __init__(self, path) -> None:
        with open(path, "rb") as dataset:
            self.dataset = pickle.load(dataset)
        self.load_data()

    def load_data(self):

        self.u_f0 = torch.from_numpy(self.dataset["u_f0"]).float()
        self.u_lo = torch.from_numpy(self.dataset["u_loudness"]).float()
        self.e_f0 = torch.from_numpy(self.dataset["e_f0"]).float()
        self.e_lo = torch.from_numpy(self.dataset["e_loudness"]).float()
        self.onsets = torch.from_numpy(self.dataset["onsets"]).float()
        self.offsets = torch.from_numpy(self.dataset["offsets"]).float()

    def get_trans_frames(self, ratio=0.1):

        trans = torch.zeros_like(self.onsets)
        frames = torch.zeros_like(self.onsets)
        note_on = 0

        for i in range(self.onsets.shape[0]):
            if self.onsets[i]:
                note_on = i
            elif self.offsets[i]:
                l = i - note_on
                l_onset = int(ratio * l)
                # update transition tensor
                s_attack, e_attack = note_on, note_on + l_onset
                trans[s_attack:e_attack] = 1
                s_release, e_release = i - l_onset, i
                trans[s_release:e_release] = 1
                #update frames tensor
                frames[e_attack:s_release] = 1

        return trans, frames

    def get_notes(self, frames, f0, lo):
        notes = []
        onsets = self.get_onsets(frames)
        for onset in onsets:
            note = {"start": None, "end": None, "f0": None, "lo": None}
            note["start"], note["end"] = onset["start"], onset["end"]
            note["f0"] = torch.mean(f0[note["start"]:note["end"]])
            note["lo"] = torch.mean(lo[note["start"]:note["end"]])

            notes.append(note)
        return notes

    def get_f0_l0_df(self):
        pitch = []
        loudness = []
        cat = []
        midi, target = self.get_all_notes()

        for note in midi:
            pitch += [note["f0"]]
            loudness += [note["lo"]]
            cat += ["midi"]

        for note in target:
            pitch += [note["f0"]]
            loudness += [note["lo"]]
            cat += ["target"]

        df = pd.DataFrame({
            "pitch": pd.Series(pitch, dtype="float32"),
            "loudness": pd.Series(loudness, dtype="float32"),
            "cat": pd.Categorical(cat),
        })

        return df

    def get_all_notes(self):
        trans, frames = self.get_trans_frames()
        midi = self.get_notes(frames, self.u_f0, self.u_lo)
        target = self.get_notes(frames, self.e_f0, self.e_lo)

        return midi, target

    def get_onsets(self, frames):
        note = {"start": None, "end": None}
        onsets = []

        for i in range(len(frames)):
            if frames[i] and note["start"] is None:
                note["start"] = i
            elif not frames[i] and note["start"] is not None:  # note turned off
                note["end"] = i
                onsets.append(note)
                note = {"start": None, "end": None}
        return onsets

    def score_pitch(self, x, y, reduction="mean"):
        y[y == 0] = 0.001
        x[x == 0] = 0.001
        d_cents = torch.abs(1200 * torch.log2(torch.abs(x / y)))
        d_cents[torch.isnan(d_cents)] = 0

        if reduction == "mean":
            return torch.mean(d_cents)
        elif reduction == "median":
            return torch.median(d_cents)
        elif reduction == "sum":
            return torch.sum(d_cents)
        else:
            print("ERROR reduction type")
            return None

    def score(self, reduction="mean"):
        trans, frames = self.get_trans_frames()
        score_trans = self.score_pitch(self.u_f0 * trans, self.e_f0 * trans,
                                       reduction)
        score_frames = self.score_pitch(self.u_f0 * frames, self.e_f0 * frames,
                                        reduction)

        return score_trans, score_frames


path = "dataset/dataset-diffusion.pickle"

analyzer = Analyzer(path)
trans, frames = analyzer.get_trans_frames()
midi, target = analyzer.get_all_notes()

df = analyzer.get_f0_l0_df()
print(df.dtypes)

# sns.set_theme(style="darkgrid")
# fig, ax = plt.subplots()
# g = sns.jointplot(x="pitch", y="loudness", data=df, hue="type", kind="kde")
# plt.show()

g = sns.jointplot(x="pitch",
                  y="loudness",
                  data=df,
                  kind="kde",
                  hue="cat",
                  alpha=.7)
plt.show()