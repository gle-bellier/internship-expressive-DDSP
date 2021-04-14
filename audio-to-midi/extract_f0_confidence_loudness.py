import os 
import os.path
from os import path
import matplotlib.pyplot as plt
import csv
import sounddevice as sd
import soundfile as sf
import numpy as np 
import librosa as li
import crepe


class Extractor:
    def __init__(self, path = "f0-confidence-loudness-files/"):
        self.path = path


    def read_file(self, file_path):
        time = []
        f0 = []
        confidence = []
        loudness = []

        with open(file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time.append(float(row["time"]))
                f0.append(float(row["f0"]))
                confidence.append(float(row["confidence"]))
                loudness.append(float(row["loudness"]))
                
        return np.array(time), np.array(f0), np.array(confidence), np.array(loudness)

    
    def write_file(self, file_path, time, f0, confidence, loudness):
        print("Writing : \n")
        print("Time shape = {}, f0 shape = {}, confidence shape = {}, loudness shape = {}".format(time.shape, f0.shape, confidence.shape, loudness.shape))
        with open(file_path, 'w') as csvfile:
            fieldnames = ["time", "f0", "confidence", "loudness"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for t in range(time.shape[0]):
               writer.writerow({"time": str(time[t]), "f0": str(f0[t]), "confidence" : str(confidence[t]), "loudness" : str(loudness[t])})

    


    def extract_loudness(self, signal, sampling_rate, block_size, n_fft=2048):
        S = li.stft(
            signal,
            n_fft=n_fft,
            hop_length=block_size,
            win_length=n_fft,
            center=True,
        )
        S = np.log(abs(S) + 1e-7)
        f = li.fft_frequencies(sampling_rate, n_fft)
        a_weight = li.A_weighting(f)

        S = S + a_weight.reshape(-1, 1)

        S = np.mean(S, 0)[..., :-1]

        return S


    def extract_time_pitch_confidence(self, signal, sampling_rate, block_size):
        f0 = crepe.predict(
            signal,
            sampling_rate,
            step_size=int(1000 * block_size / sampling_rate),
            verbose=0,
            center=True,
            viterbi=True,
        )
        return f0[0].reshape(-1)[:-1], f0[1].reshape(-1)[:-1], f0[2].reshape(-1)[:-1]






    def extract_f0_confidence_loudness(self, filename, sampling_rate, block_size):
        audio, fs = sf.read(filename, dtype='float32')   
        print("Sampling rate : ", fs)
        sampling_rate = fs
        loudness = self.extract_loudness(audio, sampling_rate, block_size)
        time, f0, confidence = self.extract_time_pitch_confidence(audio, sampling_rate, block_size)
        return time, f0, confidence, loudness 

 



    def get_time_f0_confidence_loudness(self, filename, sampling_rate, block_size, write = True):
        
        name = filename[:-4] # remove .wav
        file_path = self.path + name + "_{}_{}.csv".format(sampling_rate, block_size)

        if path.exists(file_path): # file already exists : we return the content 
            return self.read_file(file_path)

        else: # we need to extract f0 confidence loudness
            time, f0, confidence, loudness = self.extract_f0_confidence_loudness(filename, sampling_rate, block_size)


            # Check dimensions : 
            if not time.shape[0] == f0.shape[0] == confidence.shape[0] == loudness.shape[0]:
                print("!!Warning!! Shapes do not match \n")
                print("Time shape = {}, f0 shape = {}, confidence shape = {}, loudness shape = {}".format(time.shape, f0.shape, confidence.shape, loudness.shape))
                size = min(time.shape[0], f0.shape[0], confidence.shape[0], loudness.shape[0])
                print("New size : ", size)
                time, f0, confidence, loudness = time[:size], f0[:size], confidence[:size], loudness[:size] 




            if write: # we need to write the file
                self.write_file(file_path, time, f0, confidence, loudness)
            
            return time, f0, confidence, loudness

    






if __name__ == "__main__":
    filename = "violin.wav"
    sampling_rate = 48000
    block_size = 480

    ext = Extractor()
    time, f0, confidence, loudness = ext.get_time_f0_confidence_loudness(filename, sampling_rate, block_size, write=True)

    print("Time shape = {}, f0 shape = {}, confidence shape = {}, loudness shape = {}".format(time.shape, f0.shape, confidence.shape, loudness.shape))

    plt.plot(time, f0/np.max(np.abs(f0)), label = "f0")
    plt.plot(time, loudness/np.max(np.abs(loudness)), label = "Loudness")
    plt.plot(time, confidence/np.max(np.abs(confidence)), label = "Confidence")
    plt.legend()
    plt.show()
