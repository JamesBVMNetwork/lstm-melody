import tqdm
import pickle
import numpy as np
import pandas as pd
from music21 import converter, instrument, chord, note, stream
from utils import list_files_recursive


def extract_notes_from_midi(file_path):
    notes = []
    pick = None
    midi = converter.parse(file_path)
    songs = instrument.partitionByInstrument(midi)
    for part in songs.parts:
        pick = part.recurse()
        for element in pick:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
            else:
                pass
    return notes

class MelodyDataset:
    def __init__(self, config):
        self.config = config 
        assert "data_dir" in config, "data_dir is required in config for MelodyDataset"
        self.files = list_files_recursive(config["data_dir"])
        self.corpus = self._create_corpus()
        self.note_names = sorted(list(set(self.corpus)))
        self.note_to_index = dict((note, number) for number, note in enumerate(self.note_names))
        self.index_to_note = dict((number, note) for number, note in enumerate(self.note_names))

    def _create_corpus(self):
        corpus = []
        for file in tqdm.tqdm(self.files, total = len(self.files)):
            if file.endswith('.mid'):
                corpus = corpus + extract_notes_from_midi(file)
            elif file.endswith('.pickle'):
                with open(file, 'rb') as f:
                    note_list = pickle.load(f)
                    corpus = corpus + note_list
            else:
                continue
        return corpus

    def get_corpus(self):
        return self.corpus
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx: int):
        return self.corpus[idx]

    def get_vocab(self):
        return self.note_names

    def index_to_note(self, idx):
        return self.index_to_note[idx]
    
    def note_to_index(self, note):
        return self.note_to_index[note]

    def get_training_data(self):
        sequence_length = self.config.get("sequence_length", 10)
        inputs = []
        targets = []
        for i in range(0, len(self.corpus) - sequence_length):
            sequence_in = [self.note_to_index[char] for char in self.corpus[i:i + sequence_length]]
            sequence_out = self.corpus[i + sequence_length]
            inputs.append(sequence_in)
            targets.append([self.note_to_index[sequence_out]])
        inputs = np.reshape(inputs, (len(inputs), sequence_length))
        targets = np.array(targets)
        return inputs, targets

if __name__=="__main__":
    dataset = MelodyDataset({"data_dir": "data"})
    X, y = dataset.get_training_data()    

