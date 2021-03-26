import pretty_midi
from note_seq.protobuf import music_pb2
import note_seq
import numpy as np



class Converter:
    def __init__(self) -> None:
        pass
    
    def midi2note_tuple(self, midi_data):
        """ Inputs : midi data, out_file_name (optionnal)"""


    def note_tuple2midi(self, note_tuples):
        """ Inputs : note_tuple, out_file_name (optionnal)"""

    def midi2midi_like(self, midi_data, frame_rate=1000): # TODO : find another way to compute frame rate
        """ Inputs : midi file, out_file_name (optionnal)"""

        midi_like_seq = []
        n = len(midi_data.instruments)
        for instrument_data in midi_data.instruments:
            notes = instrument_data.get_piano_roll(frame_rate)
            print(notes.shape)
            buffer = np.zeros(notes.shape[0])
            for i in range(notes.shape[1]-1):
                col = notes[:,i]
                time_shift = 0 # keep track of time shifting 
                for j in range(notes.shape[0]):
                    if col[j] !=0 and buffer[j] == 0:  # note on 
                        midi_like_seq.append("NOTE_ON<{}>".format(j))
                        if time_shift > 0:
                            midi_like_seq.append("TIME_SHIFT<{}>".format(time_shift))
                            time_shift = 0

                    elif col[j] == 0 and buffer[j] != 0: # note off
                        midi_like_seq.append("NOTE_OFF<{}>".format(j))
                        time_shift = 0
                        if time_shift > 0:
                            midi_like_seq.append("TIME_SHIFT<{}>".format(time_shift))
                            time_shift = 0

                    elif col[j] != buffer[j]: # velocity change
                        midi_like_seq.append("SET_VELOCITY<{}>".format(buffer[j]))
                        time_shift = 0
                        if time_shift > 0:
                            midi_like_seq.append("TIME_SHIFT<{}>".format(time_shift))
                            time_shift = 0

                    else:
                        time_shift += int((1/frame_rate)*1000)
                buffer = notes[:,i]
        #need to close last note
        col = notes[:,-1]
        for j in range(notes.shape[0]):
            if col[j] == 0 and buffer[j] != 0:  # note on 
                midi_like_seq.append("NOTE_OFF<{}>".format(j))
                if time_shift > 0:
                    midi_like_seq.append("TIME_SHIFT<{}>".format(time_shift))
                    time_shift = 0
            else:
                time_shift += int((1/frame_rate)*1000)
        
        return midi_like_seq
                    

        











    def midi_like2midi(self, midi_like_content):
        """ Inputs : note_tuple file, out_file_name (optionnal)"""

        # SET_VELOCITY<70>
        # NOTE_ON<30>
        # TIME_SHIFT<30>
        # NOTE_OFF <50>
        current_time = 0
        velocity = 0
        notes_queue = [] # notes not yet ended (pitch, starting_time, velocity)
        sequence =  music_pb2.NoteSequence()

        for task in midi_like_content:
            if task[:7] == "SET_VEL":
                velocity = int(task[13:len(task)-1])
            
            if task[:7] == "TIME_SH":
                current_time += int(task[11:len(task)-1])
                #print("Current time ", current_time)

            if task[:7] == "NOTE_ON":
                pitch = int(task[8:len(task)-1])
                notes_queue.append((pitch, current_time, velocity))
                #print("Note ON : {} at {}ms".format(pitch, current_time))

            if task[:7] == "NOTE_OF":
                pitch = int(task[9:len(task)-1])
                for note in notes_queue:
                    if note[0] == pitch:
                        sequence.notes.add(pitch = note[0], start_time = note[1], end_time = current_time, velocity = note[2])
                        #print("New note : (pitch = {}, start_time = {}, end_time = {}, velocity = {})".format(note[0], note[1], current_time, note[2]))
                        break
        
        return sequence

