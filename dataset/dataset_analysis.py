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

    def get_all_notes(self):
        trans, frames = self.get_trans_frames()
        onsets = self.get_onsets(frames)

        midi = []
        target = []

        for onset in onsets:
            note_midi = {
                "start": None,
                "end": None,
                "f0": None,
                "lo": None,
                "diff_cents": None,
                "accurate": None
            }
            note_target = {
                "start": None,
                "end": None,
                "f0": None,
                "lo": None,
                "diff_cents": None,
                "accurate": None
            }

            # Set onsets
            note_midi["start"], note_midi["end"] = onset["start"], onset["end"]
            note_target["start"], note_target["end"] = onset["start"], onset[
                "end"]

            # MIDI
            note_midi["f0"] = torch.mean(
                self.u_f0[note_midi["start"]:note_midi["end"]])
            note_midi["lo"] = torch.mean(
                self.u_lo[note_midi["start"]:note_midi["end"]])

            # Performance

            note_target["f0"] = torch.mean(
                self.e_f0[note_target["start"]:note_target["end"]])
            note_target["lo"] = torch.mean(
                self.e_lo[note_target["start"]:note_target["end"]])

            # Compute accuracy

            d_cents, accuracy = self.accuracy(note_midi["f0"],
                                              note_target["f0"])

            note_midi["diff_cents"] = d_cents
            note_target["diff_cents"] = -d_cents
            note_midi["accurate"] = note_target["accurate"] = accuracy

            midi.append(note_midi)
            target.append(note_target)

        return midi, target

    def get_all_transitions(self):
        trans, frames = self.get_trans_frames()
        transitions = self.get_onsets(trans)

        midi = []
        target = []

        for t in transitions:
            trans_midi = {
                "start": None,
                "end": None,
                "d_f0": None,
                "d_lo": None,
                "diff_cents": None,
            }
            trans_target = {
                "start": None,
                "end": None,
                "d_f0": None,
                "d_lo": None,
                "diff_cents": None,
            }

            # Set onsets
            trans_midi["start"], trans_midi["end"] = t["start"], t["end"]
            trans_target["start"], trans_target["end"] = t["start"], t["end"]

            # MIDI
            trans_midi["d_f0"] = self.u_f0[trans_midi["end"]] - self.u_f0[
                trans_midi["start"]]
            trans_midi["d_lo"] = self.u_lo[trans_midi["end"]] - self.u_lo[
                trans_midi["start"]]

            # Performance

            trans_target["d_f0"] = self.e_f0[trans_target["end"]] - self.e_f0[
                trans_target["start"]]
            trans_target["d_lo"] = self.e_lo[trans_target["end"]] - self.e_lo[
                trans_target["start"]]

            # Compute accuracy

            d_cents = self.score_pitch(
                self.e_f0[trans_target["start"]:trans_target["end"]],
                self.u_f0[trans_target["start"]:trans_target["end"]])

            trans_midi["diff_cents"] = d_cents
            trans_target["diff_cents"] = d_cents

            midi.append(trans_midi)
            target.append(trans_target)

        return midi, target

    def accuracy(self, pitch, f0):
        d_cents = 1200 * torch.log2(torch.abs(pitch / f0))
        return d_cents, d_cents < 50

    def get_notes_df(self):
        pitch = []
        loudness = []
        cat = []
        diff_cents = []
        accuracy = []
        midi, target = self.get_all_notes()

        for note in midi:
            pitch += [note["f0"]]
            loudness += [note["lo"]]
            diff_cents += [note["diff_cents"]]
            accuracy += [note["accurate"]]
            cat += ["midi"]

        for note in target:
            pitch += [note["f0"]]
            loudness += [note["lo"]]
            diff_cents += [note["diff_cents"]]
            accuracy += [note["accurate"]]
            cat += ["target"]

        df = pd.DataFrame({
            "pitch": pd.Series(pitch, dtype="float32"),
            "loudness": pd.Series(loudness, dtype="float32"),
            "diff_cents": pd.Series(diff_cents, dtype="float32"),
            "accuracy": pd.Series(accuracy, dtype="bool"),
            "cat": pd.Categorical(cat),
        })

        return df

    def get_transitions_df(self):
        d_f0 = []
        d_lo = []
        diff_cents = []
        cat = []
        midi, target = self.get_all_transitions()

        for note in midi:
            d_f0 += [note["d_f0"]]
            d_lo += [note["d_lo"]]
            diff_cents += [note["diff_cents"]]
            cat += ["midi"]

        for note in target:
            d_f0 += [note["d_f0"]]
            d_lo += [note["d_lo"]]
            diff_cents += [note["diff_cents"]]
            cat += ["target"]

        df = pd.DataFrame({
            "d_f0": pd.Series(d_f0, dtype="float32"),
            "d_lo": pd.Series(d_lo, dtype="float32"),
            "diff_cents": pd.Series(diff_cents, dtype="float32"),
            "cat": pd.Categorical(cat)
        })

        return df

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
# trans, frames = analyzer.get_trans_frames()
# midi, target = analyzer.get_all_notes()

df = analyzer.get_transitions_df()
print(df.dtypes)

# sns.set_theme(style="darkgrid")

# g = sns.jointplot(x="pitch",
#                   y="loudness",
#                   data=df,
#                   kind="kde",
#                   hue="cat",
#                   alpha=.7)

# g = sns.jointplot(x="pitch",
#                   y="accuracy",
#                   data=df,
#                   kind="kde",
#                   hue="cat",
#                   alpha=.7)

# g = sns.jointplot(x="pitch",
#                   y="diff_cents",
#                   data=df,
#                   kind="kde",
#                   hue="cat",
#                   alpha=.7)

# g = sns.jointplot(x="loudness",
#                   y="diff_cents",
#                   data=df,
#                   kind="kde",
#                   hue="cat",
#                   alpha=.7)

g = sns.jointplot(x="d_f0",
                  y="diff_cents",
                  data=df,
                  kind="kde",
                  hue="cat",
                  alpha=.7)

plt.show()