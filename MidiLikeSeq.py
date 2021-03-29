class MidiLikeSeq:
    def __init__(self):
        self.seq = []
        self.notes_on = [] # list of pitches of notes played a this given time


    def note_on(self, pitch):
        if pitch in self.notes_on:
            print("Error : note {} is already on".format(pitch))
        else:
            self.seq.append("NOTE_ON<{}>".format(pitch))
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

    def show(self, indexes = None):
        if indexes == None:
            a, b = 0, len(self.seq)
        else:
            (a,b) = indexes
        for i in range(a,b):
            print(self.seq[i])




        