import pretty_midi
import numpy as np
import pandas as pd
import note_seq
from note_seq.protobuf import music_pb2


from MidiLikeSeq import MidiLikeSeq
from NoteTupleSeq import NoteTupleSeq


class Converter:
    def __init__(self) -> None:
        pass
    
    def midi2df(self,midi_file):
        mlseq = note_seq.midi_file_to_note_sequence(midi_file)
        pitches = []
        velocities = []
        start_times = []
        end_times = []


        for note in mlseq.notes:
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


    def df2midi_likePOLY(self,df):
        
        def insert_in_list_events(list_events, note):
            i = 0 
            while i<len(list_events) and list_events[i][0]< note[0]: # while previous notes end before
                i+=1
            list_events.insert(i, note)
             

        def select_next_move(note,list_events, current_time, velocity, seq):

            #print("DEBUG : note = {} current time = {}, list_events = {} ".format(note, current_time, list_events))


            if list_events == [] or note[2] < list_events[0][0]: # if there is not note on or if note as to start before the others end 
                if note[2]-current_time > 0: # need time shift
                    seq.time_shift(note[2]-current_time)
                current_time = note[2]
                #print("note on {} - --current time {}".format(note[0],current_time))
                insert_in_list_events(list_events, (note[3],note[0]))

                if note[1] != velocity:
                    seq.set_velocity(note[1])
                    velocity = note[1]   
                seq.note_on(note[0])
            
            else: # a note needs to end before 
                note_to_end = list_events.pop(0) # get and remove note to end
                end_time = note_to_end[0]
                if end_time-current_time > 0:
                    seq.time_shift(end_time-current_time)
                current_time = end_time
                    #print("note off {} - --current time {}".format(note_to_end[1],current_time))
                seq.note_off(note_to_end[1])
                select_next_move(note, list_events, current_time, velocity,seq)

            return current_time, list_events, velocity

        current_time = 0
        velocity = 0
        list_current_notes = [] # notes on (end_time,pitch)
        midi_like_seq = MidiLikeSeq()
        for i in range(df.shape[0]):
            note = (df.iloc[i]["Pitch"], df.iloc[i]["Velocity"], df.iloc[i]["Start time"], df.iloc[i]["End time"])
            current_time, list_current_notes, velocity = select_next_move(note, list_current_notes, current_time, velocity, midi_like_seq)
        # turn off last note:
        note_to_end = list_current_notes.pop(0) # get and remove note to end
        end_time = note_to_end[0]
        if end_time-current_time > 0:
            midi_like_seq.time_shift(end_time-current_time)
        current_time = end_time
        midi_like_seq.note_off(note_to_end[1])

        return midi_like_seq


    def df2midi_likeMONO(self, df):
        """Monophonic version"""

        current_time = 0
        velocity = 0
        midi_like_seq = MidiLikeSeq()
        for i in range(df.shape[0]):
            note = (df.iloc[i]["Pitch"], df.iloc[i]["Velocity"], df.iloc[i]["Start time"], df.iloc[i]["End time"])
            time_shift = note[2] - current_time
            if time_shift>0:
                midi_like_seq.time_shift(time_shift)
                current_time = note[2]

            if note[1] != velocity:
                midi_like_seq.set_velocity(note[1])
                velocity = note[1]
            
            midi_like_seq.note_on(note[0])
            
            duration = note[3] - note[2]
            midi_like_seq.time_shift(duration)

            midi_like_seq.note_off(note[0])

            current_time = note[3]
        
        return midi_like_seq



    def midi2midi_like(self, midi_file):
        df = self.midi2df(midi_file)
        return self.df2midi_likeMONO(df)


    def midi_like2seq(self, midi_like_content):
        """ Inputs : midi like sequence"""

        current_time = 0
        velocity = 0
        notes_queue = [] # notes not yet ended (pitch, starting_time, velocity)
        sequence =  music_pb2.NoteSequence()

        for task in midi_like_content.seq:
            if task[:7] == "SET_VEL":
                velocity = int(float(task[13:len(task)-1]))

            if task[:7] == "TIME_SH":
                current_time += float(task[11:len(task)-1])/1000

            if task[:7] == "NOTE_ON":
                pitch = int(float(task[8:len(task)-1]))
                notes_queue.append((pitch, current_time, velocity))

            if task[:7] == "NOTE_OF":
                pitch = int(float(task[9:len(task)-1]))
                for note in notes_queue:
                    if note[0] == pitch:
                        sequence.notes.add(pitch = note[0], start_time = note[1], end_time = current_time, velocity = note[2])                        
                        break
                notes_queue.remove(note)
        
        return sequence



    def df2note_tuple(self, df):
        """ Inputs : sorted data frame of notes"""

        def time_shift2ticks(time_shift):
            major_gap = 10/13 # 13 ticks for 10s
            minor_gap = major_gap/77 # 77 minor ticks between two major ticks

            major_ticks = time_shift // major_gap
            minor_ticks = (time_shift % major_gap) // minor_gap

            return (major_ticks, minor_ticks)

        
        def duration2ticks(duration):
            major_gap = 10/25 # 25 ticks for 10s
            minor_gap = major_gap/40 # 40 minor ticks between two major ticks

            major_ticks = duration // major_gap
            minor_ticks = (duration % major_gap) // minor_gap

            return (major_ticks, minor_ticks)

        previous_note_start_time = 0  # keep track of the beginning of the previous note (initialized at 0)
        note_tuple_seq = NoteTupleSeq()


        for i in range(df.shape[0]):
            pitch = df.iloc[i]["Pitch"]
            velocity = df.iloc[i]["Velocity"]
            duration = df.iloc[i]["End time"] - df.iloc[i]["Start time"]
            time_shift = df.iloc[i]["Start time"] -previous_note_start_time
            previous_note_start_time = df.iloc[i]["Start time"] # update last note

            ts_M, ts_m = time_shift2ticks(time_shift)
            d_M, d_m = duration2ticks(duration)
            note_tuple = (int(ts_M), int(ts_m), int(pitch), int(velocity), int(d_M), int(d_m))
            note_tuple_seq.add_note(note_tuple)
        
        return note_tuple_seq


    def midi2note_tuple(self, midi_file):
        df = self.midi2df(midi_file)
        return self.df2note_tuple(df)



    def note_tuple2seq(self, note_tuple_seq):
        """ Input : note_tuple
            Output : Seq"""

        sequence =  music_pb2.NoteSequence()

        def time_shift_ticks2time(ts_M, ts_m):
            return (10/13) * ts_M + (10/(13*77)) * ts_m
            
        
        def duration_ticks2time(d_M, d_m):
            return (10/25) * d_M + (10/(25*40)) * d_m

        last_note_starting_time = 0
        for t in note_tuple_seq.seq:
            p = t[2]
            v = t[3]
            ts_M, ts_m = t[0], t[1]
            d_M, d_m = t[4], t[5]

            start_t = last_note_starting_time + time_shift_ticks2time(ts_M, ts_m)
            end_t = start_t + duration_ticks2time(d_M, d_m)
            sequence.notes.add(pitch = p, start_time = start_t, end_time = end_t, velocity = v)  
            last_note_starting_time = start_t                      

        return sequence


    def midi2time_f0_loudness(self, midi_data, frame_rate = 16000, times_needed = None):
        
        instrument_data = midi_data.instruments[0]
        if times_needed is None:
            notes = instrument_data.get_piano_roll(frame_rate)
            times = np.array([i/frame_rate for i in range(notes.shape[1])])
        else:
            notes = instrument_data.get_piano_roll(times = times_needed)
            times = times_needed
        
        pitches, loudness = self.extract_f0_loudness(notes)
        

        return times, pitches, loudness

    def extract_f0_loudness(self, notes):
        pitches = np.argmax(notes, axis = 0)
        loudness = np.transpose(np.max(notes, axis = 0))
        return pitches, loudness

