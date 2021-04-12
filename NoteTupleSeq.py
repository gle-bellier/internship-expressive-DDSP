import numpy as np

class NoteTupleSeq:
    def __init__(self):
        self.seq = []
        self.notes_on = [] # list of pitches of notes played a this given time
        self.duration = 0 
        self.pitch = None
        self.loudness = None


    def add_note(self, note):
        self.seq.append(note)


    def show(self, indexes = None):
        if indexes == None:
            a, b = 0, len(self.seq)
        else:
            (a,b) = indexes
        for i in range(a,b):
            print(self.seq[i])

    def __repr__(self) -> str:
        s = ""
        for task in self.seq:
            t = [str(elt) for elt in task]
            s+= "("+','.join(t)+")"+"\n"
        return s

    def __eq__(self, o: object) -> bool:
        if len(self.seq)!=len(o.seq):
            return False
        else:
            for i in range(len(self.seq)):
                if self.seq[i]!=o.seq[i]:
                    return False
        return True

    def save(self, filename):
        file = open(filename, "w")
        for task in self.seq:
            t = [str(elt) for elt in task]
            s = "("+','.join(t)+")"
            file.write(s+"\n")
        file.close()

    def load(self, filename):
        file = open(filename, "r")
        line = file.readline()
        while line != "":
            line = *(int(elt) for elt in line[1:-2].split(",")),
            self.seq.append(line)
            line = file.readline()
        file.close()

    def get_f0_loudness_time(self, frame_rate, pitch_unit = "MIDI"):
        """ Input : frame rate (Hz), pitch unit (MIDI or HERTZ) 
        Output : Pitch (float array), Loudness (float array), Loudness (float array) 
        Extract f0 and loudness from monophonic tracks"""


        def time_shift_ticks2time(ts_M, ts_m):
                return (10/13) * ts_M + (10/(13*77)) * ts_m
                    
        def duration_ticks2time(d_M, d_m):
            return (10/25) * d_M + (10/(25*40)) * d_m

        def write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time):
            self.pitch[i_previous_event_time:i_current_time] = current_pitch * note_ON
            self.loudness[i_previous_event_time:i_current_time] = current_loudness *note_ON
        
        # compute track duration
        total_duration = 0
        previous_note_duration = 0 #since time shift is calculated since the beginning of the previous note and note the end
        for note in self.seq:
            ts = time_shift_ticks2time(note[0], note[1])
            d = duration_ticks2time(note[4], note[5])
            total_duration += (ts + d - previous_note_duration)
            previous_note_duration = d
        self.duration = total_duration
        print(total_duration)

        self.pitch = np.zeros(int(self.duration * frame_rate))
        self.loudness = np.zeros(int(self.duration * frame_rate))


        i_current_time = 0
        i_previous_event_time = 0 
        previous_note_duration = 0 #since time shift is calculated since the beginning of the previous note and note the end

        for note in self.seq:
            time_shift = time_shift_ticks2time(note[0], note[1]) - previous_note_duration
            duration = duration_ticks2time(note[4], note[5])
            previous_note_duration = duration
            i_previous_event_time = i_current_time
            i_current_time += int(time_shift // (1/frame_rate))
            write_events(note[2], note[3], False, i_current_time, i_previous_event_time)
            i_previous_event_time = i_current_time
            i_current_time += int(duration // (1/frame_rate))
            write_events(note[2], note[3], True, i_current_time, i_previous_event_time)

        
        # remove tail : 
        i = 1
        while self.pitch[-i]==0:
            i+=1

        print("Tail length : ", i)
        self.pitch = self.pitch[:-i]
        self.loudness = self.loudness[:-i]

        # convert midi pitch to hz
        if pitch_unit == "HERTZ":
            self.pitch = np.power(2, (self.pitch-69)/12)*440    



        t = np.arange(self.pitch.shape[0])/frame_rate

        return self.pitch, self.loudness, t
        

