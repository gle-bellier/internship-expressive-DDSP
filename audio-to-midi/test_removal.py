import librosa as li
import numpy as np
import matplotlib.pyplot as plt
import resampy

def get_window(signal, index, width):
    a, b = index-width//2, index+width//2
    if a<0: a = 0
    if b>signal.shape[0]:n = signal.shape[0]-1
    return signal[a:b]



sampling_rate = 16000
audio, sr = li.load("vn_32_1_Fugue.wav")
resampy.resample(audio, sr, sampling_rate)

window_size_t = 0.5
window_size_n = int(window_size_t*sampling_rate)

m = np.array([np.mean(get_window(np.abs(audio), i, window_size_n)) for i in range(0,audio.shape[0],window_size_n)])

t = np.array(range(audio.shape[0]))/sampling_rate
t_win = np.array(range(m.shape[0]))/sampling_rate*window_size_n

plt.plot(t, audio)
plt.plot(t_win, m)
plt.show()



