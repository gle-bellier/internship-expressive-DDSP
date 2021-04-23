import crepe
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li


from madmom.features import onsets

import note_seq
from note_seq.protobuf import music_pb2

from extract_f0_confidence_loudness import Extractor


class Audio2MidiConverter:
    def __init__(self, filename):
        self.filename = filename
        self.sampling_rate = None
        self.min_note_length = None



    def dB2midi(self, loudness, global_peak = None, global_min = None):
        if global_peak == None:
            global_peak = np.max(loudness)
        if global_min == None:
            global_min = np.min(loudness)
        
        l = loudness-global_min
        L =  127*np.abs(l)/np.abs(global_peak-global_min)
        return L 

    def compute_dv(self, confidence):
        conf_delay = np.concatenate((np.zeros((1)), confidence))
        confidence_pad = np.concatenate((confidence, np.zeros((1))))
        dv = confidence_pad - conf_delay
        return dv[:-1]

    def get_loudness_changes(self, loudness, threshold):
        dl = self.compute_dv(loudness)
        dln = dl/np.max(np.abs(dl))
        positiv = dln > threshold
        negativ = dln < -threshold
        return negativ, positiv

    def get_confidence_changes(self, confidence, threshold):
        dv = self.compute_dv(confidence)
        positiv = dv > threshold
        negativ = dv < -threshold
        return negativ, positiv


    def get_midi_pitch_changes(self, freq):
        pitch = li.core.hz_to_midi(freq)
        pitch_delay = np.concatenate((np.zeros((1)), pitch))
        pitch_pad = np.concatenate((pitch, np.zeros((1))))
        dp = np.abs(pitch_pad - pitch_delay) > 0
        return dp[:-1]



    def get_note_with_pitch_loudness(self, note, frequency, loudness):
        note_w_pitch = {"on": None, "off": None, "pitch": None, "loudness": None}
        note_w_pitch["on"] = note["on"]
        note_w_pitch["off"] = note["off"]
        note_w_pitch["pitch"] = int(li.core.hz_to_midi(np.mean(frequency[note["on"]: note["off"]])))
        note_w_pitch["loudness"] = int(np.mean(loudness[note["on"]: note["off"]])) # TODO : Change for normalisation of loudness
        return note_w_pitch

    


    def get_window(self, comp, index,  h_window_length):
        a = index - h_window_length
        b = index + h_window_length
        if a<0: a = 0 
        if b>=comp.shape[0]: b =comp.shape[0]-1
        a, b = int(a), int(b)
        return comp[a:b]

        

    def local_AND(self, support, comp, h_window_length):
        onsets = np.zeros(support.shape[0])


        for i in range(support.shape[0]):
            if support[i]: # need to check comp
                window = self.get_window(comp, i, h_window_length)
                onsets[i] = np.sum(window) > 0 # There is at least one other offset detected in the window
        return onsets


    def join_notes(self, notes, time, min_duration = 0.1):

        joined_notes = []

        for i in range(len(notes)-1):
            note, next_note = notes[i], notes[i+1]
            if note["pitch"]==next_note["pitch"] and time[next_note["on"]]-time[note["off"]]<=min_duration: # notes of the same pitch and really close 
                new_note = {"on": None, "off": None, "pitch": None, "loudness": None}
                new_note["on"] = note["on"]
                new_note["off"] = next_note["off"] # notes are joined
                new_note["pitch"] = note["pitch"]
                new_note["loudness"] = (note["loudness"] + next_note["loudness"])//2
                joined_notes.append(new_note)
            else:
                joined_notes.append(note)

        return joined_notes



    def process(self, sampling_rate = 48000, block_size = 480, threshold_confidence = 0.15, threshold_loudness = 0.20, min_note_length = 0.01,  verbose = False):
        self.sampling_rate = sampling_rate
        self.min_note_length = min_note_length



        ext = Extractor()
        time, frequency, confidence, loudness = ext.get_time_f0_confidence_loudness(self.filename, sampling_rate, block_size, write=True)
        loudness = self.dB2midi(loudness)

        neg_conf_changes, pos_conf_changes = self.get_confidence_changes(confidence, threshold_confidence)
        midi_pitch_changes = self.get_midi_pitch_changes(frequency)
        neg_loud_changes, pos_loud_changes = self.get_loudness_changes(loudness, threshold_loudness)

        

        conf_onsets = np.logical_or(pos_conf_changes, neg_conf_changes)
        loud_onsets = np.logical_or(pos_loud_changes, neg_loud_changes)
        pitch_onsets = midi_pitch_changes



        # all_onsets = np.logical_or(pos_conf_changes, neg_conf_changes) # May be change to AND
        # all_onsets = np.logical_or(all_onsets, midi_pitch_changes) # May be change to OR 
        # all_onsets = np.logical_or(all_onsets, loud_onsets)   

        t_h_window_length = 0.005 # in s
        h_window_length = np.ceil(t_h_window_length * self.sampling_rate) # in samples

        neg_changes = self.local_AND(neg_conf_changes, neg_loud_changes, h_window_length)
        pos_changes = self.local_AND(pos_conf_changes, pos_loud_changes, h_window_length)

        all_onsets = self.local_AND(pos_changes, neg_changes, h_window_length)



        # Notes creation


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

            elif neg_changes[t]:
                if current_note["on"] is not None:
                    current_note["off"] = t
                    notes.append(current_note)
                    current_note = {"on": None, "off": None}







        # remove short notes

        notes = [note for note in notes if note["off"] - note["on"] >= min_note_length]

        # add pitch information
        notes_with_pitch_loudness = [self.get_note_with_pitch_loudness(note, frequency, loudness) for note in notes]

        #join successive notes of same pitch (carefully) 
        notes = self.join_notes(notes_with_pitch_loudness, time)


        # writing note in sequence
        sequence =  music_pb2.NoteSequence()

        for note in notes:
            sequence.notes.add(pitch = note["pitch"], start_time = time[note["on"]], end_time = time[note["off"]], velocity = note["loudness"])                        
            
        return sequence

        
        
    def process_madmom(self, sampling_rate = 48000, block_size = 480, threshold_confidence = 0.15, threshold_madmom = 0.20, min_note_length = 0.01,  verbose = False):
        self.sampling_rate = sampling_rate
        self.min_note_length = min_note_length



        ext = Extractor()
        time, frequency, confidence, loudness = ext.get_time_f0_confidence_loudness(self.filename, sampling_rate, block_size, write=True)
        neg_changes, _ = self.get_confidence_changes(confidence, threshold_confidence)
        loudness = self.dB2midi(loudness)


        proc = onsets.OnsetPeakPickingProcessor(threshold = threshold_madmom, fps=100)
        act = onsets.RNNOnsetProcessor()(self.filename)

        onset_times = proc(act)
        # create vector:
        all_onsets = np.zeros_like(time)
        i_onset = 0 
        for i in range(time.shape[0]-1):
            onset = onset_times[i_onset]
            if time[i]<= onset and onset<time[i+1]: # onset in segment
                all_onsets[i] = 1 # add onset
                if i_onset+1 < len(onset_times):
                    i_onset += 1 # we need to set next onset (if there is one)
            
    
        # Notes creation


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

            elif neg_changes[t]:
                if current_note["on"] is not None:
                    current_note["off"] = t
                    notes.append(current_note)
                    current_note = {"on": None, "off": None}



        if verbose:
            span = time.shape[0]//8
            middle = time.shape[0]//2
            a, b = middle - span, middle + span
   

            ax3 = plt.subplot(224)
            ax3.plot(time[a:b], frequency[a:b]/np.max(frequency), label = "Normalized f0")
            ax3.plot(time[a:b], all_onsets[a:b], label = "Onsets")
            ax3.legend()
            ax3.set_title('Madmom')

            plt.legend()
            plt.show()





        if verbose:
            span = time.shape[0]//8
            middle = time.shape[0]//2
            a, b = middle - span, middle + span

            ax1 = plt.subplot(212)        
            ax1.plot(time[a:b], frequency[a:b]/np.max(frequency), label = "Normalized f0")
            ax1.plot(time[a:b], np.ones((b-a))*threshold_confidence)

            ax1.plot(time[a:b], np.abs(self.compute_dv(confidence[a:b])), label = "Dv Confidence")
            ax1.set_title("f0 confidence")

            ax2 = plt.subplot(221)
       
            ax2.plot(time[a:b], frequency[a:b]/np.max(frequency), label = "Normalized f0")
            ax2.plot(time[a:b], all_onsets[a:b], label = "Onsets")
            ax2.set_title('Onsets')
            plt.legend()
            plt.show()





        # remove short notes

        notes = [note for note in notes if note["off"] - note["on"] >= min_note_length]

        # add pitch information
        notes_with_pitch_loudness = [self.get_note_with_pitch_loudness(note, frequency, loudness) for note in notes]
        notes = self.join_notes(notes_with_pitch_loudness, time)

        # writing note in sequence
        sequence =  music_pb2.NoteSequence()
        
        for note in notes:
            sequence.notes.add(pitch = note["pitch"], start_time = time[note["on"]], end_time = time[note["off"]], velocity = note["loudness"])                        
            
        return sequence 








    



if __name__ == "__main__":
    #filename = "violin.wav"
    filename = "flute.wav"
    save_path = "midi-generated-files/"
    threshold_confidence = 0.2
    threshold_loudness = 0.3
    threshold_madmom = 0.3   
    a2m = Audio2MidiConverter(filename)

    seq_midi = a2m.process(sampling_rate = 48000, block_size = 480, threshold_confidence = threshold_confidence, threshold_loudness = threshold_loudness, verbose = True)
    #seq_midi = a2m.process_madmom(sampling_rate = 48000, block_size = 480, threshold_confidence = threshold_confidence, threshold_madmom = threshold_madmom, verbose = True)
    note_seq.sequence_proto_to_midi_file(seq_midi, save_path + filename[:-4] + "(from-audio)-thC{}-thL{}.mid".format(threshold_confidence, threshold_loudness))
    
