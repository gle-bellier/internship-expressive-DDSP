import numpy as np

class MidiLikeSeq:
    def __init__(self):
        self.seq = []
        self.notes_on = [] # list of pitches of notes played a this given time
        self.duration = 0


    def note_on(self, pitch):
        if pitch in self.notes_on:
            print("Error : note {} is already on".format(pitch))
        else:
            self.seq.append("NOTE_ON<{}>".format(int(pitch)))
            self.notes_on.append(pitch)

    def note_off(self, pitch):
        if pitch not in self.notes_on:
            print("Error : can not turn off unexisting note (pitch {}).".format(pitch))
        else:
            self.notes_on.remove(pitch)
            self.seq.append("NOTE_OFF<{}>".format(pitch))
    
    def set_velocity(self, v):
        self.seq.append("SET_VELOCITY<{}>".format(v))

    def time_shift(self, delay):
        self.seq.append("TIME_SHIFT<{}>".format(delay))
        self.duration += delay

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
            s+=task
            s+="\n"
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
            file.write(task+"\n")
        file.close()

    def load(self, filename):
        file = open(filename, "r")
        line = file.readline()
        while line != "":
            self.seq.append(line[:-1])
            line = file.readline()
        file.close()

    def get_f0_loudness(self, frame_rate):
        """ extract f0 and loudness for monophonic tracks"""


        def write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time, pitch, loudness):
                pitch[i_previous_event_time:i_current_time] = current_pitch * note_ON
                loudness[i_previous_event_time:i_current_time] = current_loudness *note_ON
                return pitch, loudness

        current_pitch = 0
        current_loudness = 0
        i_current_time = 0
        i_previous_event_time = 0 
        note_ON = False
        
        pitch = np.zeros(int(self.duration * frame_rate))
        loudness = np.zeros(int(self.duration * frame_rate))
        
        for task in self.seq:
            if task[:7] == "SET_VEL":
                pitch, loudness = write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time, pitch, loudness)
                current_loudness = int(float(task[13:len(task)-1]))

            if task[:7] == "TIME_SH":
                time_shift = float(task[11:len(task)-1])
                i_previous_event_time = i_current_time
                i_current_time += time_shift // frame_rate

            if task[:7] == "NOTE_ON":
                pitch, loudness = write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time, pitch, loudness)
                note_ON = True
                current_pitch = int(float(task[8:len(task)-1]))

            if task[:7] == "NOTE_OF":
                pitch, loudness = write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time, pitch, loudness)
                note_ON = False
        
        return pitch, loudness




        