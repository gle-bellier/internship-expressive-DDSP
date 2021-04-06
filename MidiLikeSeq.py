import numpy as np


class MidiLikeSeq:
    def __init__(self):
        """ Class for Midi-Like sequences. """

        self.seq = []
        self.notes_on = [] # list of pitches of notes played a this given time
        self.duration = 0
        self.pitch = None
        self.loudness = None


    def note_on(self, pitch):
        """ Input : pitch [0,127] midi norm, Output : None 
        Append note on task in the sequence """

        if pitch in self.notes_on:
            print("Error : note {} is already on".format(pitch))
        else:
            self.seq.append("NOTE_ON<{}>".format(int(pitch)))
            self.notes_on.append(pitch)


    def note_off(self, pitch):
        """ Input : pitch [0,127] midi norm, Output : None 
        Append note off task in the sequence """

        if pitch not in self.notes_on:
            print("Error : can not turn off unexisting note (pitch {}).".format(pitch))
        else:
            self.notes_on.remove(pitch)
            self.seq.append("NOTE_OFF<{}>".format(pitch))
    

    def set_velocity(self, v):
        """ Input : velocity [0,127] midi norm, Output : None 
        Append set velocity task in the sequence """

        self.seq.append("SET_VELOCITY<{}>".format(v))


    def time_shift(self, delay):
        """ Input : time shift (s), Output : None 
        Append note time shift (ms) in the sequence """

        self.seq.append("TIME_SHIFT<{}>".format(delay*1000))
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

    def get_f0_loudness_time(self, frame_rate):
        """ extract f0 and loudness for monophonic tracks"""


        def write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time):
                #print("Index previous time {}: , Index Current time {}".format(i_previous_event_time, i_current_time))
                #print("Pitch : ", current_pitch)
                #print("Loudness: ", current_loudness)
                #print("Note On : ", note_ON)
                self.pitch[i_previous_event_time:i_current_time] = current_pitch * note_ON
                self.loudness[i_previous_event_time:i_current_time] = current_loudness *note_ON

        current_pitch = 0
        current_loudness = 0
        i_current_time = 0
        i_previous_event_time = 0 
        note_ON = False
        
        self.pitch = np.zeros(int(self.duration * frame_rate))
        self.loudness = np.zeros(int(self.duration * frame_rate))

        
        for task in self.seq:
            if task[:7] == "SET_VEL":
                write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time)
                current_loudness = int(float(task[13:len(task)-1]))

            if task[:7] == "TIME_SH":
                time_shift = float(task[11:len(task)-1])/1000
                i_previous_event_time = i_current_time
                i_current_time += int(time_shift // (1/frame_rate))

            if task[:7] == "NOTE_ON":
                write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time)
                note_ON = True
                current_pitch = int(float(task[8:len(task)-1]))
                i_previous_event_time = i_current_time

            if task[:7] == "NOTE_OF":
                write_events(current_pitch, current_loudness, note_ON, i_current_time, i_previous_event_time)
                note_ON = False
                i_previous_event_time = i_current_time
            
            t = np.arange(self.pitch.shape[0])/frame_rate
        
        return self.pitch, self.loudness, t




        