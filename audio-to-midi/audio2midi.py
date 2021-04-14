import crepe
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li

import note_seq
from note_seq.protobuf import music_pb2

from extract_f0_confidence_loudness import Extractor


class Audio2MidiConverter:
    def __init__(self, filename):
        self.filename = filename
        self.sampling_rate = None
        self.min_note_length = None



    def hertz2midi(self, frequency):
        return np.ceil(12 * np.log2(frequency/440)) + 69


    def dB2midi(self, loudness, global_peak = None, global_min = None):
        if global_peak == None:
            global_peak = np.max(loudness)
        if global_min == None:
            global_min = np.min(loudness)
        
        l = loudness-global_min
        print("l : ", l)
        L =  127*np.abs(l)/np.abs(global_peak-global_min)
        print("Loudness : ", L )
        return L 

    def compute_dv(self, confidence):
        conf_delay = np.concatenate((np.zeros((1)), confidence))
        confidence_pad = np.concatenate((confidence, np.zeros((1))))
        dv = confidence_pad - conf_delay
        return dv[:-1]


    def get_confidence_changes(self, confidence, threshold):
        dv = self.compute_dv(confidence)
        positiv = dv > threshold
        negativ = dv < -threshold
        return negativ, positiv


    def get_midi_pitch_changes(self, freq):
        pitch = self.hertz2midi(freq)
        pitch_delay = np.concatenate((np.zeros((1)), pitch))
        pitch_pad = np.concatenate((pitch, np.zeros((1))))
        dp = np.abs(pitch_pad - pitch_delay) > 0
        return dp[:-1]



    def get_note_with_pitch_loudness(self, note, frequency, loudness):
        note_w_pitch = {"on": None, "off": None, "pitch": None, "loudness": None}
        note_w_pitch["on"] = note["on"]
        note_w_pitch["off"] = note["off"]
        note_w_pitch["pitch"] = int(self.hertz2midi(np.mean(frequency[note["on"]: note["off"]])))
        note_w_pitch["loudness"] = int(np.mean(loudness[note["on"]: note["off"]])) # TODO : Change for normalisation of loudness
        return note_w_pitch

    


    def process(self, sampling_rate = 48000, block_size = 480, threshold = 0.15, min_note_length = 0.01):
        self.sampling_rate = sampling_rate
        self.min_note_length = min_note_length



        ext = Extractor()
        time, frequency, confidence, loudness = ext.get_time_f0_confidence_loudness(filename, sampling_rate, block_size, write=True)
        

        pos_conf_changes, neg_conf_changes = self.get_confidence_changes(confidence, threshold)
        midi_pitch_changes = self.get_midi_pitch_changes(frequency)

        all_onsets = np.logical_or(pos_conf_changes, neg_conf_changes) # May be change to AND
        all_onsets = np.logical_or(all_onsets, midi_pitch_changes) # May be change to OR 


        notes = []
        current_note = {"on": None, "off": None}
        t_length = all_onsets.shape[0]

        for t in range(t_length):
            if all_onsets[t]:
                if current_note["on"] is not None:
                    current_note["off"] = t
                    notes.append(current_note)
                    current_note = {"on": None, "off": None}
                current_note["on"] = t

            elif neg_conf_changes[t]:
                if current_note["on"] is not None:
                    current_note["off"] = t
                    notes.append(current_note)
                    current_note = {"on": None, "off": None}

        # remove short notes

        notes = [note for note in notes if note["off"] - note["on"] >= min_note_length]
        loudness = self.dB2midi(loudness)


        notes_with_pitch = [self.get_note_with_pitch_loudness(note, frequency, loudness) for note in notes]


        print(notes_with_pitch)
        sequence =  music_pb2.NoteSequence()

        for note in notes_with_pitch:
            sequence.notes.add(pitch = note["pitch"], start_time = time[note["on"]], end_time = time[note["off"]], velocity = note["loudness"])                        
            
        return sequence

        
        
        








    



if __name__ == "__main__":
    filename = "violin.wav"

    threshold = 0.10
    a2m = Audio2MidiConverter(filename)
    seq_midi = a2m.process(sampling_rate = 48000, block_size = 480, threshold = threshold)


    note_seq.sequence_proto_to_midi_file(seq_midi, filename[:-4] + "(from-audio)-th{}.mid".format(threshold))
    
