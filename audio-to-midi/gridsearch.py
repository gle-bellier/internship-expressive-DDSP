from audio2midi import Audio2MidiConverter
from extract_f0_confidence_loudness import Extractor
import sys
sys.path.insert(0,'..')
from midiConverter import Converter


import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li

import pandas as pd
import pretty_midi as pm
import note_seq
from note_seq.protobuf import music_pb2
import matplotlib.pyplot as plt


def hertz2midi(frequency):
    return 12 * np.log2(frequency/440) + 69



filename = "violin.wav"
sampling_rate = 48000
block_size = 480 


ext = Extractor()
time, frequency, _, loudness = ext.get_time_f0_confidence_loudness(filename, sampling_rate, block_size, write=True)

print("Track Duration : ", time[-1])
print("Vectors length : ", time.shape[0])



filename = "violin.wav"
save_path = "midi-generated-files/"
threshold_confidence = 0.08
threshold_loudness = 0.2   
a2m = Audio2MidiConverter(filename)
seq_midi = a2m.process(sampling_rate = 48000, block_size = 480, threshold_confidence = threshold_confidence, threshold_loudness = threshold_loudness, verbose = False)

save_file = save_path + filename[:-4] + "(from-audio)-thC{}-thL{}.mid".format(threshold_confidence, threshold_loudness)
note_seq.sequence_proto_to_midi_file(seq_midi, save_file)



print("Start converting : ", save_file)

c = Converter()
midi_data = pm.PrettyMIDI(save_file)

print("->   File loaded.")
time_gen, frequency_gen, loudness_gen = c.midi2time_f0_loudness(midi_data, sampling_rate/block_size)


frequencyMidi = hertz2midi(frequency)


print("Track Duration : ", time_gen[-1])
print("Vectors length : ", time_gen.shape[0])
new_freq_gen = np.concatenate((frequency_gen, np.zeros(time.shape[0]-time_gen.shape[0])))

print("New array shape : ", new_freq_gen.shape)


plt.plot(time, frequencyMidi)
plt.plot(time, new_freq_gen)
plt.show()


print("Difference")

plt.plot(time, np.abs(frequencyMidi-new_freq_gen))
plt.show()

