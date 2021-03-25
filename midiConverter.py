import pretty_midi
from note_seq.protobuf import music_pb2
import note_seq



class Converter:
    def __init__(self) -> None:
        pass
    
    def midi2note_tuple(self, midi_data):
        """ Inputs : midi data, out_file_name (optionnal)"""


    def note_tuple2midi(self, note_tuples):
        """ Inputs : note_tuple, out_file_name (optionnal)"""

    def midi2midi_like(self, midi_file):
        """ Inputs : midi file, out_file_name (optionnal)"""


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






















# import pretty_midi
# # Create a PrettyMIDI object
# cello_c_chord = pretty_midi.PrettyMIDI()
# # Create an Instrument instance for a cello instrument
# cello_program = pretty_midi.instrument_name_to_program('Cello')
# cello = pretty_midi.Instrument(program=cello_program)
# # Iterate over note names, which will be converted to note number later
# for note_name in ['C5', 'E5', 'G5']:
#     # Retrieve the MIDI note number for this note name
#     note_number = pretty_midi.note_name_to_number(note_name)
#     # Create a Note instance, starting at 0s and ending at .5s
#     note = pretty_midi.Note(
#         velocity=100, pitch=note_number, start=0, end=.5)
#     # Add it to our cello instrument
#     cello.notes.append(note)
# # Add the cello instrument to the PrettyMIDI object
# cello_c_chord.instruments.append(cello)
# # Write out the MIDI data
# cello_c_chord.write('cello-C-chord.mid')