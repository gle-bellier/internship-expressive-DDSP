from audio2midi import Audio2MidiConverter
from extract_f0_confidence_loudness import Extractor
from txt2contours import Txt2Contours


import glob
import sys
sys.path.insert(0,'..')
from midiConverter import Converter

from tqdm import tqdm
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li
import seaborn as sns

import pandas as pd
import pretty_midi as pm
import note_seq
from note_seq.protobuf import music_pb2
import matplotlib.pyplot as plt






class Eval:
    def __init__(self):
        pass

    def evaluate(self, dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, verbose=False):

        # From text file
        
        ext = Extractor()
        time_wav, frequency_wav, _, loudness_wav = ext.get_time_f0_confidence_loudness(dataset_path, wav_file, sampling_rate, block_size, write=True)
        # From midi file : 
        c = Converter()
        midi_data = pm.PrettyMIDI(dataset_path + midi_file)
        time_gen, pitch_gen, loudness_gen = c.midi2time_f0_loudness(midi_data, times_needed=time_wav)# sampling_rate/block_size, None)# time_wav)

        frequency_gen = li.core.midi_to_hz(pitch_gen)
        loudness_gen = loudness_gen / np.max(loudness_gen)
        # want to erase really quiet notes
        loudness_threshold = 0.20
        frequency_gen = frequency_gen * (loudness_gen>loudness_threshold)


        diff_f0 = np.abs(frequency_gen - frequency_wav)
        diff_loudness = np.abs(loudness_wav - loudness_gen)

        # # compute difference and score:

        score = np.mean(diff_f0) + np.mean(diff_loudness)


        if verbose:
            ax1 = plt.subplot(221)        
            ax1.plot(time_wav, frequency_wav, label = "wav")
            ax1.plot(time_gen, frequency_gen, label = "midi" )
            ax1.set_title("f0 comparison")
            ax1.legend()

            ax2 = plt.subplot(222)
            ax2.plot(time_wav, loudness_wav, label = "wav")
            ax2.plot(time_gen, loudness_gen/np.max(loudness_gen), label = "midi" )
            ax2.set_title('Loudness comparison')
            ax2.legend()


            plt.title("{} and {}".format(midi_file, wav_file))
            plt.legend()
            plt.show()

        return score



if __name__ == '__main__':

    dataset_path = "dataset-midi-wav/"
    filenames =[file[len(dataset_path):-4] for file in glob.glob(dataset_path + "*.mid")]

    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            print("Filename : ", filename)
            midi_file = filename + ".mid"
            wav_file = filename + ".wav"
            e = Eval()
            score = e.evaluate(dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, verbose=True)
            print("Total score : ", score)
            pbar.update(1)
        

