from audio2midi import Audio2MidiConverter
from extract_f0_confidence_loudness import Extractor
from txt2contours import Txt2Contours


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




sampling_rate = 48000
block_size = 480


filename = "vn_01_Jupiter"
text_filename = filename + ".txt"
audio_filename = filename + ".wav"


ext = Extractor()
time, f0, _, loudness = ext.get_time_f0_confidence_loudness(audio_filename, sampling_rate, block_size, write=True)

t2c = Txt2Contours()
time_text, frequency_text, loudness_text = t2c.process(text_filename, sampling_rate/block_size)



ax1 = plt.subplot(211)        
ax1.plot(time_text, frequency_text, label = "text")
ax1.plot(time, f0, label = "wav" )
ax1.set_title("f0 comparison")

ax2 = plt.subplot(212)
ax2.plot(time_text, loudness_text, label = "text")
ax2.plot(time, loudness, label = "wav" )
ax2.set_title('Loudness comparison')




plt.legend()
plt.show()
