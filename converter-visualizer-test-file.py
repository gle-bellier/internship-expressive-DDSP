from midiConverter import Converter
from Visualizer import Visualizer

import pretty_midi as pm
from note_seq.protobuf import music_pb2
import note_seq
import glob

if __name__ == "__main__":
    c = Converter()
    mlseq = ["SET_VELOCITY<30>",
            "NOTE_ON<50>",
            "TIME_SHIFT<80>",
            "NOTE_ON<62>",
            "TIME_SHIFT<30>",
            "NOTE_OFF<50>",
            "SET_VELOCITY<160>",
            "NOTE_OFF<62>"
    ]
    mseq = c.midi_like2midi(mlseq)
    note_seq.sequence_proto_to_midi_file(mseq, 'midi-like2midi.mid')

      
    midi_data = pm.PrettyMIDI('bassline.mid')
    v = Visualizer(midi_data)
    v.show_midi_notes()
    v.show_f0_velocity()
    