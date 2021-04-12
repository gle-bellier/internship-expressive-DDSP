import crepe
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt




class Audio2MidiConverter:
    def __init__(self, filename, sampling_rate, min_note_length):
        self.filename = filename
        self.sampling_rate = sampling_rate
        self.min_note_length = min_note_length
