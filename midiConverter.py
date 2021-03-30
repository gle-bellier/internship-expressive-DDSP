import pretty_midi
from note_seq.protobuf import music_pb2
import note_seq
import numpy as np
import pandas as pd
from MidiLikeSeq import MidiLikeSeq



class Converter:
    def __init__(self) -> None:
        pass
    
    def midi2df(self,midi_file):
        mlseq = note_seq.midi_file_to_note_sequence('bassline.mid')
        pitches = []
        velocities = []
        start_times = []
        end_times = []


        for note in mlseq.notes:
            print(note)
            pitches.append(note.pitch)
            velocities.append(note.velocity)
            start_times.append(note.start_time)
            end_times.append(note.end_time)


        pd_pitches = pd.Series(pitches)
        pd_velocities = pd.Series(velocities)
        pd_start_times = pd.Series(start_times)
        pd_end_times = pd.Series(end_times)


        df = pd.DataFrame({"Pitch": pd_pitches, "Velocity": pd_velocities, "Start time": pd_start_times, "End time": pd_end_times})
        df.sort_values(by=["Start time","Pitch"])
        return df


    def df2midi_like(self,df):

        def select_next_move(note,list_events,current_time,velocity,seq):
            if list_events == [] or note[2]<list_events[0][0]:
                seq.time_shift(note[2]-current_time)
                current_time = note[2]
                list_events.append((note[3],note[0]))
                
                if note[1] != velocity:
                    seq.set_velocity(note[1])    
                seq.note_on(note[0])
            
            else: # a note needs to end before 
                note_to_end = list_events.pop(0) # get and remove note to end
                end_time = note_to_end[0]
                seq.time_shift(end_time-current_time)
                current_time = end_time
                seq.note_off(note_to_end[1])
                # checking next note : 
                select_next_move(note, list_events, current_time, velocity,seq)

        current_time = 0
        velocity = 0
        list_current_notes = [] # notes on (end_time,pitch)
        midi_like_seq = MidiLikeSeq()
        for i in range(df.shape[0]):
            print(df.iloc[i])
            note = (df.iloc[i]["Pitch"], df.iloc[i]["Velocity"], df.iloc[i]["Start time"], df.iloc[i]["End time"])
            select_next_move(note, list_current_notes, current_time, velocity, midi_like_seq)
        print(midi_like_seq)
        return midi_like_seq


    def midi2midi_like(self, midi_file):
        df = self.midi2df(midi_file)
        return self.df2midi_like(df)


    def midi_like2midi(self, midi_like_content):
        """ Inputs : note_tuple file, out_file_name (optionnal)"""

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



    def midi2note_tuple(self, midi_data):
        """ Inputs : midi data, out_file_name (optionnal)"""


    def note_tuple2midi(self, note_tuples):
        """ Inputs : note_tuple, out_file_name (optionnal)"""

