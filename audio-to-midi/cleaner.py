import pandas as pd
import pretty_midi as pm
from note_seq.protobuf import music_pb2
import note_seq
from tqdm import tqdm
import glob



class Cleaner:

    def __init__(self):
        pass

    def check(self, midi_file, epsi_error=0.01):
        mlseq = note_seq.midi_file_to_note_sequence(midi_file)
        previous_note = mlseq.notes[0]

        for note in mlseq.notes[1:]:
            if note.start_time<previous_note.end_time -epsi_error   :
                print("Note : pitch {}, start {}, end {} ".format(note.pitch, note.start_time, note.end_time))
                print("Next note : pitch {}, start {}, end {}".format(previous_note.pitch, previous_note.start_time, previous_note.end_time ))
                return False
            previous_note = note

        return True


if __name__ == '__main__':
    
    dataset_path = "dataset-midi-wav/"
    filenames =[file[len(dataset_path):-4] for file in glob.glob(dataset_path + "*.mid")]

    with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
            midi_file = filename + ".mid"
            wav_file = filename + ".wav"
            c = Cleaner()
            print("{} file is monophonic : {}".format(midi_file, c.check(dataset_path + midi_file)))
            pbar.update(1)

            
            