import torch as torch
import io as io
import scipy.io.wavfile as wav
import numpy as np


ddsp = torch.jit.load("ddsp_debug_pretrained.ts")





f0 = torch.randn(1,8000,1)
l0 = torch.randn(1,8000,1)

audio = ddsp(f0, l0).detach().squeeze().numpy()

print(audio.shape)


sampling_rate = 16000
filename = "essai1.wav"

wav.write(filename, sampling_rate, audio.astype(np.int16))




