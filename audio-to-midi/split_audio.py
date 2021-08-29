import numpy as np
import soundfile as sf

PATH = "audio-to-midi/vn_01_Jupiter.wav"
LENGTH = 20 * 60

i = 0
for block in sf.blocks(PATH, blocksize=44100 * LENGTH):
    save_path = PATH.split(".")
    save_path = "{}{}.wav".format(save_path[0], i)

    sf.write(save_path, block, 44100)
    i += 1
