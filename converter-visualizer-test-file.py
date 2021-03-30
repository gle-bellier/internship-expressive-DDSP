from midiConverter import Converter
from Visualizer import Visualizer
from MidiLikeSeq import MidiLikeSeq


import pandas as pd
import pretty_midi as pm
from note_seq.protobuf import music_pb2
import note_seq
import glob

if __name__ == "__main__":
    c = Converter()
    mlseq = ["SET_VELOCITY<30>",
            "NOTE_ON<50>",
            "TIME_SHIFT<80>",
            "NOTE_OFF<50>",
            "SET_VELOCITY<103>",
            "NOTE_ON<42>",
            "TIME_SHIFT<30>",
            "NOTE_OFF<42>",
            "SET_VELOCITY<60>",
            "NOTE_ON<40>",
            "TIME_SHIFT<30>",
            "NOTE_OFF<40>"]
    
    
    

    # midi_data = pm.PrettyMIDI('bassline.mid')
    # mlseq = c.midi2midi_like(midi_data)

    # print("Midi like seq : ")
    # print(mlseq)
    

#     mseq = c.midi_like2midi(mlseq)
#     print("Midi seq : ")    
#     print(mseq)
#     note_seq.sequence_proto_to_midi_file(mseq, 'bassline-reconstruct.mid')

      
#     midi_data = pm.PrettyMIDI('bassline-reconstruct.mid')
#     v = Visualizer(midi_data)
#     v.show_midi_notes(DEBUG = True)
#     v.show_f0_velocity()
    seq = MidiLikeSeq()
    seq.note_on(63)
    seq.time_shift(40)
    seq.set_velocity(70)
    seq.note_off(63)
    seq.note_on(40)
    seq.time_shift(120)
    seq.note_off(40)
    seq.note_on(63)
    seq.time_shift(40)
    seq.set_velocity(70)
    seq.note_off(63)
    seq.show((3,5))

    mlseq = note_seq.midi_file_to_note_sequence('bassline.mid')
    print(mlseq)



    pitches = []
    velocities = []
    start_times = []
    end_time = []


    for note in mlseq.notes:
        print(note)
        pitches.append(note.pitch)
        velocities.append(note.velocity)
        start_times.append(note.start_time)
        end_time.append(note.end_time)
        

    