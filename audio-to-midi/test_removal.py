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
    
    def get_window(self, signal, index, width):
        a, b = index-width//2, index+width//2
        if a<0: a = 0
        if b>signal.shape[0]:n = signal.shape[0]-1
        return signal[a:b]


    def onset_offset(self, dm, max_silence):
        l = []
        silence = {"on" : 0, "off": None}
        for i in range(len(dm)-1):

            if dm[i]==True and dm[i+1]==False:
                silence["on"]=i
                

            if dm[i]==False and dm[i+1]==True:
                silence["off"]=i
                if silence["off"] - silence["on"]>max_silence:
                    l.append(silence)
                silence = {"on" : None, "off": None}
        return l

                


    def evaluate(self, dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration = 1, verbose=False):

        # From text file
        
        ext = Extractor()
        time_wav, frequency_wav, _, loudness_wav = ext.get_time_f0_confidence_loudness(dataset_path, wav_file, sampling_rate, block_size, write=True)
        # loudness mapping : 
        loudness_wav = self.dB2midi(loudness_wav)
        
        
        
        # From midi file : 
        c = Converter()
        midi_data = pm.PrettyMIDI(dataset_path + midi_file)
        time_gen, pitch_gen, loudness_gen = c.midi2time_f0_loudness(midi_data, times_needed=time_wav)# sampling_rate/block_size, None)# time_wav)

        frequency_gen = li.core.midi_to_hz(pitch_gen)
        # want to erase really quiet notes
        loudness_threshold = 0.20
        frequency_gen = frequency_gen * (loudness_gen / np.max(loudness_gen)>loudness_threshold)

        threshold = 20
        m = np.array([np.mean(self.get_window(loudness_wav, i, sampling_rate//100)) for i in range(loudness_wav.shape[0])])
        tm = (m>threshold)
        

        
        max_silence_length =  max_silence_duration * (sampling_rate/block_size)

        indexes = self.onset_offset(tm, max_silence_length)
        #print(indexes)
        onsets = np.zeros_like(loudness_wav)
        for idx in indexes:
            onsets[idx["on"]]=1.0
            onsets[idx["off"]]=-1.0



        # loudness mapping : 
        

        diff_f0 = np.abs(frequency_gen - frequency_wav)
        diff_loudness = np.abs(loudness_wav - loudness_gen)


        # # compute difference and score:

        score = np.mean(diff_f0) + np.mean(diff_loudness)


        if verbose:

            ax1 = plt.subplot(221)
            ax1.plot(time_wav, loudness_wav, label = "wav")
            ax1.plot(time_wav, m, label = "m" )
            ax1.set_title('Loudness comparison')
            ax1.legend()

            ax2 = plt.subplot(222)
            ax2.plot(time_wav, loudness_wav/np.max(loudness_wav), label = "wav")
            ax2.plot(time_wav, onsets, label = "m" )
            ax2.set_title('Loudness comparison')
            ax2.legend()

            plt.title("{} and {}".format(midi_file, wav_file))
            plt.legend()
            plt.show()

        return score
    

    def dB2midi(self, loudness, global_peak = None, global_min = None):
        loudness = li.core.db_to_amplitude(loudness)
        if global_peak == None:
            global_peak = np.max(loudness)
        if global_min == None:
            global_min = np.min(loudness)
        
        l = loudness-global_min
        L =  127*np.abs(l)/np.abs(global_peak-global_min)
        return L 



if __name__ == '__main__':
    
    dataset_path = "dataset-midi-wav/"
    filenames =[file[len(dataset_path):-4] for file in glob.glob(dataset_path + "*.mid")]

    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            midi_file = filename + ".mid"
            wav_file = filename + ".wav"
            e = Eval()
            score = e.evaluate(dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration=3, verbose=True)
            pbar.update(1)
            
            
