from numpy.lib.function_base import extract
import pretty_midi as pm
import matplotlib.pyplot as plt
import glob
import numpy as np 



class Visualizer:
    def __init__(self, midi_data,name = "midi file"):
        self.midi_data = midi_data
        self.name = name

    def show_midi_notes(self):
        n = len(self.midi_data.instruments)
        fig, axs = plt.subplots(n,1, figsize=(15, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.1)
        if n>1:
            axs = axs.ravel()
            for j in range(n):
                notes = self.midi_data.instruments[j].get_piano_roll()
                axs[j].imshow(notes,cmap="cividis",aspect='auto')
                axs[j].set_title("Track : {} | Instrument : {}".format(self.name,self.midi_data.instruments[j].name))
                axs[j].set_ylim((0,128))
                axs[j].set_xlabel("Time")
                axs[j].set_ylabel("Pitch")
            plt.show()
        else:
            notes = self.midi_data.instruments[0].get_piano_roll()
            plt.imshow(notes,cmap="cividis",aspect='auto')
            plt.title("Track : {} | Instrument : {}".format(self.name,self.midi_data.instruments[0].name))
            plt.ylim((0,128))
            plt.xlabel("Time")
            plt.ylabel("Pitch")
            plt.show()

    def show_f0_velocity(self,frame_rate = 250):
        n = len(self.midi_data.instruments)
        for instrument_data in self.midi_data.instruments:
            notes = instrument_data.get_piano_roll(frame_rate)
            print(notes.shape)
            pitches, loudness = self.extract_f0_loudness(notes)

    def extract_f0_loudness(self, notes):
        pitches = np.argmax(notes, axis = 0)
        loudness = np.max(notes, axis = 0)
        return pitches, loudness







