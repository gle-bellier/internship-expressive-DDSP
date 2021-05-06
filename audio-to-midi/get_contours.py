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

        
    def get_not_silence(self, indexes, times):
        l = []
        loud = [0, None]
        for idx in indexes:
            loud[1]=idx["on"]
            l.append(loud)
            loud = [idx["off"], None]
        loud[1]=len(times)
        l.append(loud)
        if l[0]==[0,0]:
            l = l[1:]
        return l

    def get_notes_loudness(self, loudness, onsets):
        notes_loudness = np.zeros_like(loudness)
        for i in range(len(onsets)-1):
            a, b = onsets[i], onsets[i+1]
            if a!=b:
                notes_loudness[a:b] = np.max(loudness[a:b])
            else:
                notes_loudness[a] = loudness[a]
        return notes_loudness 
    

    def get_onsets(self, midi_file, frame_rate):
        seq = note_seq.midi_file_to_note_sequence(midi_file)
        onsets = []
        for note in seq.notes:
            onsets.append(int(note.start_time*frame_rate))
            onsets.append(int(note.end_time*frame_rate))

        return onsets


    def dB2midi(self, loudness, global_peak = None, global_min = None):
        loudness = li.core.db_to_amplitude(loudness)
        if global_peak == None:
            global_peak = np.max(loudness)
        if global_min == None:
            global_min = np.min(loudness)
        
        l = loudness-global_min
        L =  127*np.abs(l)/np.abs(global_peak-global_min)
        return L 



    def evaluate(self, dataset_path, midi_file, wav_file, sampling_rate=16000, block_size=160, max_silence_duration = 3, verbose=False):

        # From text file
        
        ext = Extractor()
        time_wav, frequency_wav, _, loudness_wav = ext.get_time_f0_confidence_loudness(dataset_path, wav_file, sampling_rate, block_size, write=True)
        # loudness mapping : 
        loudness_wav = self.dB2midi(loudness_wav)
        
        
        
        # From midi file : 
        c = Converter()
        midi_data = pm.PrettyMIDI(dataset_path + midi_file)
        time_gen, pitch_gen, loudness_gen = c.midi2time_f0_loudness(midi_data, times_needed=time_wav)# sampling_rate/block_size, None)# time_wav)
        
        midi_onsets = self.get_onsets(dataset_path + midi_file, sampling_rate//block_size)
        loudness_gen = self.get_notes_loudness(loudness_wav, midi_onsets)


        frequency_gen = li.core.midi_to_hz(pitch_gen)
        # want to erase really quiet notes
        loudness_threshold = 0.20
        #frequency_gen = frequency_gen * (loudness_gen / np.max(loudness_gen)>loudness_threshold)

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


        parts_index = self.get_not_silence(indexes, time_wav)
        # loudness mapping : 
        

        diff_f0 = np.abs(frequency_gen - frequency_wav)
        diff_loudness = np.abs(loudness_wav - loudness_gen)


        # # compute difference and score:

        #score = np.mean(diff_f0) + np.mean(diff_loudness)
        time_list = []
        frequency_gen_list = []
        loudness_gen_list = []
        frequency_wav_list = []
        loudness_wav_list = []

        n = len(parts_index)
        for elt in parts_index:
            start, end = elt
            time_list.append(time_wav[start:end])

            frequency_wav_list.append(frequency_wav[start:end])
            loudness_wav_list.append(loudness_wav[start:end])

            frequency_gen_list.append(frequency_gen[start:end])
            loudness_gen_list.append(loudness_gen[start:end])


        if verbose:
            ax1 = plt.subplot((n+2)*100+10 + 1)
            ax1.plot(time_wav, loudness_wav, label = "wav")
            ax1.plot(time_wav, loudness_gen, label = "midi" )
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.title("Loudness comparison {}".format(wav_file))

            for i in range(n):
                ax1 = plt.subplot((n+2)*100+10 + i+2)
                ax1.plot(time_list[i], loudness_wav_list[i], label = "wav")
                ax1.plot(time_list[i], loudness_gen_list[i], label = "midi" )
                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()


            ax1 = plt.subplot((n+2)*100+10 + 1)
            ax1.plot(time_wav, frequency_wav, label = "wav")
            ax1.plot(time_wav, frequency_gen, label = "midi" )
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title("Frequency comparison {}".format(wav_file))

            for i in range(n):
                ax1 = plt.subplot((n+2)*100+10 + i+2)
                ax1.plot(time_list[i], frequency_wav_list[i], label = "wav")
                ax1.plot(time_list[i], frequency_gen_list[i], label = "midi" )
                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        return frequency_wav_list, loudness_wav_list, frequency_gen_list, loudness_gen_list
    


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
            break

            
            
