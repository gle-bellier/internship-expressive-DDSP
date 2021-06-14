from extract_f0_confidence_loudness import Extractor

import glob
import sys
from midiConverter import Converter

import csv
from tqdm import tqdm
import os
import numpy as np
# import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li
import seaborn as sns

import pandas as pd
import pretty_midi as pm
import note_seq
from note_seq.protobuf import music_pb2
import matplotlib.pyplot as plt


class ContoursGetter:
    def __init__(self):
        pass

    def get_window(self, signal, index, width):
        a, b = index - width // 2, index + width // 2
        if a < 0: a = 0
        if b > signal.shape[0]: n = signal.shape[0] - 1
        return signal[a:b]

    def onset_offset(self, dm, max_silence):
        l = []
        silence = {"on": 0, "off": None}
        for i in range(len(dm) - 1):

            if dm[i] == True and dm[i + 1] == False:
                silence["on"] = i

            if dm[i] == False and dm[i + 1] == True:
                silence["off"] = i
                if silence["off"] - silence["on"] > max_silence:
                    l.append(silence)
                silence = {"on": None, "off": None}
        return l

    def get_not_silence(self, indexes, times):
        l = []
        loud = [0, None]
        for idx in indexes:
            loud[1] = idx["on"]
            l.append(loud)
            loud = [idx["off"], None]
        loud[1] = len(times)
        l.append(loud)
        if l[0] == [0, 0]:
            l = l[1:]
        return l

    def get_notes_loudness(self, loudness, onsets):
        notes_loudness = np.zeros_like(loudness)
        previous_i = 0
        for i in range(len(onsets)):
            a, b = onsets[i]

            if previous_i != a:
                notes_loudness[previous_i:a] = np.mean(loudness[previous_i:a])
            else:
                notes_loudness[previous_i] = loudness[previous_i]

            if a != b:
                notes_loudness[a:b] = np.max(loudness[a:b])
            else:
                notes_loudness[a] = loudness[a]

            previous_i = b

        return notes_loudness

    def get_freq_mean(self, frequencies, onsets):
        freq_mean = np.zeros_like(frequencies)
        previous_i = 0
        for i in range(len(onsets)):
            a, b = onsets[i]
            if previous_i != a:
                freq_mean[previous_i:a] = np.mean(frequencies[previous_i:a])
            else:
                freq_mean[previous_i] = frequencies[previous_i]
            if a != b:
                freq_mean[a:b] = np.mean(frequencies[a:b])
            else:
                freq_mean[a] = frequencies[a]

            previous_i = b

        return freq_mean

    def get_onsets(self, midi_file, frame_rate):
        seq = note_seq.midi_file_to_note_sequence(midi_file)
        onsets = []
        for note in seq.notes:
            onsets.append((int(note.start_time * frame_rate),
                           int(note.end_time * frame_rate)))

        return onsets

    def get_contours(self,
                     dataset_path,
                     midi_file,
                     wav_file,
                     sampling_rate=16000,
                     block_size=160,
                     max_silence_duration=3,
                     verbose=False):

        # From text file

        ext = Extractor()
        time_wav, frequency_wav, _, loudness_wav = ext.get_time_f0_confidence_loudness(
            dataset_path, wav_file, sampling_rate, block_size, write=True)

        # From midi file :
        c = Converter()

        midi_data = pm.PrettyMIDI(dataset_path + midi_file)
        time_gen, pitch_midi, _ = c.midi2time_f0_loudness(
            midi_data, times_needed=time_wav)
        frequency_midi = li.core.midi_to_hz(pitch_midi)

        midi_onsets = self.get_onsets(dataset_path + midi_file,
                                      sampling_rate // block_size)
        loudness_midi = self.get_notes_loudness(loudness_wav, midi_onsets)

        frequency_wav_means = self.get_freq_mean(frequency_wav, midi_onsets)

        threshold = -5.5
        m = np.array([
            np.mean(self.get_window(loudness_wav, i, sampling_rate // 100))
            for i in range(loudness_wav.shape[0])
        ])
        tm = (m > threshold)

        max_silence_length = max_silence_duration * (sampling_rate //
                                                     block_size)

        indexes = self.onset_offset(tm, max_silence_length)
        #print(indexes)
        onsets = np.zeros_like(loudness_wav)
        for idx in indexes:
            onsets[idx["on"]] = 1.0
            onsets[idx["off"]] = -1.0

        parts_index = self.get_not_silence(indexes, time_wav)

        diff_f0 = np.abs(frequency_midi - frequency_wav)
        diff_loudness = np.abs(loudness_wav - loudness_midi)

        # # compute difference and score:

        #score = np.mean(diff_f0) + np.mean(diff_loudness)

        frequency_midi_array = np.empty(0)
        loudness_midi_array = np.empty(0)
        frequency_wav_array = np.empty(0)
        loudness_wav_array = np.empty(0)
        frequency_wav_means_array = np.empty(0)

        silence_duration = 0.5  # duration of silence we keep at each cut.
        silence_length = silence_duration * (sampling_rate // block_size)

        n = len(parts_index)
        for elt in parts_index:
            start, end = elt

            # need to check if begining or end

            if start - silence_length < 0:
                start = 0
            if end + silence_length > len(time_wav) - 1:
                end = len(time_wav) - 1

            frequency_midi_array = np.concatenate(
                (frequency_midi_array, frequency_midi[start:end]))
            loudness_midi_array = np.concatenate(
                (loudness_midi_array, loudness_midi[start:end]))

            frequency_wav_array = np.concatenate(
                (frequency_wav_array, frequency_wav[start:end]))
            loudness_wav_array = np.concatenate(
                (loudness_wav_array, loudness_wav[start:end]))

            frequency_wav_means_array = np.concatenate(
                (frequency_wav_means_array, frequency_wav_means[start:end]))

        if verbose:

            plt.plot(loudness_wav, label="gen")
            plt.plot(loudness_midi, label="midi")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Loudness comparison {}".format(wav_file))
            plt.show()

            plt.plot(frequency_wav_array, label="wav")
            plt.plot(frequency_midi_array, label="midi")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Frequency comparison {}".format(wav_file))
            plt.show()

            # plt.plot(frequency_wav_stddev_array, label="std dev")
            # plt.plot(frequency_wav_means_array /
            #          np.max(frequency_wav_means_array),
            #          label="pitch")
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.title("Frequency comparison {}".format(wav_file))
            # plt.show()

        return frequency_midi_array, loudness_midi_array, frequency_wav_array, loudness_wav_array, frequency_wav_means


if __name__ == '__main__':

    dataset_path = "dataset-article/dataset-midi-wav/"
    filenames = [
        file[len(dataset_path):-4]
        for file in glob.glob(dataset_path + "*.mid")
    ]
    print(filenames)
    duration = 0

    u_f0 = np.empty(0)
    u_loudness = np.empty(0)
    e_f0 = np.empty(0)
    e_loudness = np.empty(0)
    e_f0_mean = np.empty(0)
    e_f0_stddev = np.empty(0)

    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            midi_file = filename + ".mid"
            wav_file = filename + ".wav"
            g = ContoursGetter()
            u_f0_track, u_loudness_track, e_f0_track, e_loudness_track, e_f0_mean_track = g.get_contours(
                dataset_path,
                midi_file,
                wav_file,
                sampling_rate=16000,
                block_size=160,
                max_silence_duration=3,
                verbose=False)

            u_f0 = np.concatenate((u_f0, u_f0_track))
            u_loudness = np.concatenate((u_loudness, u_loudness_track))
            e_f0 = np.concatenate((e_f0, e_f0_track))
            e_loudness = np.concatenate((e_loudness, e_loudness_track))

            e_f0_mean = np.concatenate((e_f0_mean, e_f0_mean_track))

            pbar.update(1)
        e_f0_stddev = (1200 * np.log2(e_f0 / u_f0)).clip(-50, 50)

        print("Writing : \n")

        with open("dataset/contours-article.csv", 'w') as csvfile:
            fieldnames = [
                "u_f0", "u_loudness", "e_f0", "e_loudness", "e_f0_mean",
                "e_f0_stddev"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for t in range(u_f0.shape[0]):
                writer.writerow({
                    "u_f0": str(u_f0[t]),
                    "u_loudness": str(u_loudness[t]),
                    "e_f0": str(e_f0[t]),
                    "e_loudness": str(e_loudness[t]),
                    "e_f0_mean": str(e_f0_mean[t]),
                    "e_f0_stddev": str(e_f0_stddev[t])
                })
