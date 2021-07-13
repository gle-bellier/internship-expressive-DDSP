import torch
import pickle
from utils import *
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self, path) -> None:
        with open(path, "rb") as dataset:
            self.dataset = pickle.load(dataset)
        self.load_data()

    def load_data(self):

        self.u_f0 = torch.from_numpy(self.dataset["u_f0"]).long()
        self.u_lo = torch.from_numpy(self.dataset["u_loudness"]).float()
        self.e_f0 = torch.from_numpy(self.dataset["e_f0"]).long()
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

    def get_notes(self, frames):
        note = {"start": None, "end": None}
        l_notes = []

        for i in range(len(frames)):
            if frames[i] and note["start"] is None:
                note["start"] = i
            elif not frames[i] and note["start"] is not None:  # note turned off
                note["end"] = i
                l_notes.append(note)
                note = {"start": None, "end": None}

        return l_notes


path = "dataset/dataset-article.pickle"

analyzer = Analyzer(path)
trans, frames = analyzer.get_trans_frames()
l_notes = analyzer.get_notes(frames)