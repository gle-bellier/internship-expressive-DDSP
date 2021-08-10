from midiConverter import Converter

import csv
from tqdm import tqdm
import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
# import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li
#import seaborn as sns

import pandas as pd
import pretty_midi as pm
import note_seq
from note_seq.protobuf import music_pb2
import matplotlib.pyplot as plt
from extract_f0_confidence_loudness import Extractor


def get_onsets(midi_file, frame_rate):
    seq = note_seq.midi_file_to_note_sequence(midi_file)
    onsets = []
    for note in seq.notes:
        onsets.append((int(note.start_time * frame_rate),
                       int(note.end_time * frame_rate)))

    return onsets


def get_events(onsets, shape):
    events = np.zeros(shape)
    for elt in onsets:
        events[elt[0]] = 1  # onset
        if elt[1] < shape[0]:
            events[elt[1]] = -1  # offset
    return events


def get_midi_lo(path):

    ref_path = "dmitry-sinkovsky-plays-jsbachs-partita-in-e-major_1.wav"
    # From midi file :
    c = Converter()

    ext = Extractor()

    sampling_rate = 16000
    block_size = 160
    time_wav, f0_wav, _, lo_wav = ext.get_time_f0_confidence_loudness(
        "violin/", ref_path, sampling_rate, block_size, write=True)

    # From midi file :
    c = Converter()

    midi_data = pm.PrettyMIDI(path)
    time_gen, p, lo = c.midi2time_f0_loudness(midi_data, times_needed=time_wav)
    f0 = li.core.midi_to_hz(p)

    lo = lo.reshape(-1, 1)
    lo_wav = lo_wav.reshape(-1, 1)

    lo_rescale = QuantileTransformer().fit_transform(lo)
    scaler = QuantileTransformer().fit(lo_wav)
    lo_rescale = scaler.inverse_transform(lo_rescale)

    # Computing onsets/ offsets

    midi_onsets = get_onsets(path, sampling_rate // block_size)
    events = get_events(midi_onsets, f0_wav.shape)

    return f0, lo_rescale, events


files = ["dmitry-sinkovsky-plays-jsbachs-partita-in-e-major_1.mid"]
path = "violin/"
f0, lo, events = get_midi_lo(path + files[1])
plt.plot(lo)
plt.show()

plt.plot(events * 1000)
plt.plot(f0)
plt.show()
