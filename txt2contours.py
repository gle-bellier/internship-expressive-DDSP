import numpy as np
import matplotlib.pyplot as plt
import os


class Txt2Contours:
    def __init__(self):
        pass

    def get_list_events(self, filename):
        file = open(filename, "r")
        line = file.readline()

        list_events = []
        while line != "":
            line = *(float(elt) for elt in line[:-2].split("\t\t")),
            list_events.append(line)
            line = file.readline()
        file.close()
        return list_events


    def process(self, filename, sample_rate=16000):
        list_events = self.get_list_events(filename)

        # Compute track duration : 
        last_note = list_events[-1]
        duration = last_note[0] + last_note[2]

        # create time vector : 

        time = np.arange(0, duration, 1/sample_rate)
        f0 = np.zeros_like(time)
        loudness = np.zeros_like(time)
        i_note = 0
        note = list_events[i_note]
        onset = note[0]
        offset = note[0]+note[2]
        pitch = note[1]
        for i in range(time.shape[0]):
            if offset<time[i] and i_note+1<len(list_events):
                i_note += 1
                note = list_events[i_note]
                onset = note[0]
                offset = note[0]+note[2]
                pitch = note[1]

            if time[i]<= onset and offset>time[i]:
                f0[i] = pitch
                loudness[i] = 1
            else:
                f0[i] = 0

        return time, f0, loudness







if __name__ == '__main__':
    filename = "test.txt"
    t2c = Txt2Contours()
    time, f0, loudness = t2c.process(filename)
    
    plt.plot(time[10000:400000], f0[10000:400000]/np.max(f0), label = "Frequency")
    plt.plot(time[10000:400000], loudness[10000:400000], label = "Loudness")
    plt.show()
