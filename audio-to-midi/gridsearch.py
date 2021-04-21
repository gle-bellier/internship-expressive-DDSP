from audio2midi import Audio2MidiConverter
from extract_f0_confidence_loudness import Extractor
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





class GridSearch:
    def __init__(self, filename, sampling_rate = 48000, block_size = 480):
        self.filename = filename
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.save_path = None


    def set_save_path(self, path):
        self.save_path 

    def hertz2midi(self, frequency):
        return 12 * np.log2(frequency/440) + 69


    def get_result(self, thresholds1, thresholds2, madmom = True, verbose = False, keep_best_file = False):

        # get original f0 and loudness contours
        ext = Extractor()
        time_or, frequency_or, _, loudness_or = ext.get_time_f0_confidence_loudness(filename, sampling_rate, block_size, write=True)
        if verbose:
            print("Track Duration : ", time_or[-1])
            print("Vectors length : ", time_or.shape[0])


        confidence_length = thresholds1.shape[0]
        loudness_length = thresholds2.shape[0]


        results = np.zeros((confidence_length, loudness_length))
        
        with tqdm(total=confidence_length*loudness_length) as pbar:

            for i in range(confidence_length):
                for j in range(loudness_length):

                    a2m = Audio2MidiConverter(self.filename)    
                    if madmom:
                        threshold_confidence, threshold_madmom = thresholds1[i], thresholds2[j]
                        seq_midi = a2m.process_madmom(sampling_rate = self.sampling_rate, block_size = self.block_size, threshold_confidence = threshold_confidence, threshold_madmom = threshold_madmom, verbose = False)
                        save_file = save_path + filename[:-4] + "(from-audio)-thC{}-thM{}.mid".format(threshold_confidence, threshold_madmom)
                    else:
                        threshold_confidence, threshold_loudness = thresholds1[i], thresholds2[j]
                        seq_midi = a2m.process(sampling_rate = self.sampling_rate, block_size = self.block_size, threshold_confidence = threshold_confidence, threshold_loudness = threshold_loudness, verbose = False)
                        save_file = save_path + filename[:-4] + "(from-audio)-thC{}-thL{}.mid".format(threshold_confidence, threshold_loudness)
                        
                    note_seq.sequence_proto_to_midi_file(seq_midi, save_file)


                    if verbose:
                        print("Start converting : ", save_file)

                    c = Converter()
                    midi_data = pm.PrettyMIDI(save_file)

                    if verbose:
                        print("->   File loaded.")

                    time_gen, frequency_gen, loudness_gen = c.midi2time_f0_loudness(midi_data, sampling_rate/block_size)
                    frequencyMidi = self.hertz2midi(frequency_or)

                    # 0 padding:
                    new_freq_gen = np.concatenate((frequency_gen, np.zeros(time_or.shape[0]-time_gen.shape[0])))

                    # compute difference and score:

                    diff = np.abs(frequencyMidi-new_freq_gen)
                    score = np.mean(diff)

                    if verbose:
                        plt.plot(time_or, frequencyMidi)
                        plt.plot(time_or, new_freq_gen)
                        plt.show()

                        plt.plot(time_or, diff)
                        plt.show()
                        print("Loss : ", score)
                    
                    results[i, j] = score
                    
                    # update progress bar
                    pbar.update(1)

        self.results = results

        # clean files
        self.clean(keep_best_file, save_path + filename[:-4], results, thresholds1, thresholds2)

        return results

    def clean(self, keep_best_file, path_name, results, thresholds1, thresholds2, madmom = True):
        idx = np.argmin(results)
        a, b = idx // results.shape[1], idx % results.shape[1]
        for i in range(thresholds1.shape[0]):
            for j in range(thresholds2.shape[0]):
                if i==a and j==b and keep_best_file: # need to delete this file
                    continue

                if madmom:
                    path = path_name + "(from-audio)-thC{}-thM{}.mid".format(thresholds2[i], thresholds2[j])
                else:
                    path = path_name + "(from-audio)-thC{}-thL{}.mid".format(thresholds2[i], thresholds2[j])
                
                os.remove(path)





            




















filename = "violin.wav"
save_path = "midi-generated-files/"
sampling_rate = 48000
block_size = 480 


### FIRST EXPERIMENT ###

gs = GridSearch(filename, sampling_rate, block_size)
gs.set_save_path(save_path)

# define thresholds ranges : 
number_thresholds = 5
thresholds2 = np.linspace(0.01, 0.4, number_thresholds)
thresholds1 = np.linspace(0.01, 0.4, number_thresholds)

# get results :
rslt = gs.get_result(thresholds1, thresholds2, madmom=True, verbose=False, keep_best_file=True)

# print results : 

rslt = np.log(rslt)
rslt = rslt/np.max(rslt)


sns.heatmap(rslt)

plt.xticks(plt.xticks()[0], labels=np.round(thresholds1, 3))
plt.yticks(plt.yticks()[0], labels=np.round(thresholds2, 3))
plt.xlabel("Confidence threshold")
plt.ylabel("Loudness threshold")
plt.title("Grid Search for Audio2Midi")
plt.legend()
plt.show()


