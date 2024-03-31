import numpy as np
import argparse
import tensorflow as tf
import os
import json
from music21 import instrument, note, stream
import pandas as pd

# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.
MELODY_SIZE = 130

SEQUENCE_LENGTH = 10

def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = df.duration.diff(-1) * -1 * 0.25  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[['code','duration']]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = note.Rest() # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using a trained model')
    parser.add_argument('--model-dir', type=str, default='model', help='Directory to load model')
    parser.add_argument('--output-path', type=str, default='output.mid', help='Path to save generated music')
    return parser.parse_args()


def load_model_from_ckpt(ckpt):
    return tf.keras.models.load_model(ckpt)



def generate_melody(input_notes, vocab, model, seq_length = SEQUENCE_LENGTH, to_generate = 10):
    input_notes = input_notes[-seq_length: ]
    for i in range(seq_length - len(input_notes)):
        input_notes.insert(0, np.random.choice(vocab))
    prediction_output = []

    temperature = 1.0
    for i in range(to_generate):
        prediction_input = np.array(input_notes).reshape(1, seq_length, 1) / MELODY_SIZE
        prediction_logits = model.predict(prediction_input)

        prediction_logits = prediction_logits / temperature

        predicted_ids = tf.random.categorical(prediction_logits, num_samples=1)
        prediction = tf.squeeze(predicted_ids, axis=-1)
        # print(prediction)

        prediction_output.append(vocab[prediction[0]])
        input_notes = input_notes[1:] + [vocab[prediction[0]]]

    return prediction_output
    
def create_midi(prediction_output, output_file='test_output.mid'):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    
    midi_stream = noteArrayToStream(prediction_output)
    midi_stream.write('midi', fp=output_file)

if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    checkpoint_path = os.path.join(model_dir, 'model.h5')
    model_config_path = os.path.join(model_dir, 'model.json')
    model = load_model_from_ckpt(checkpoint_path)
    model.summary()
    with open(model_config_path, 'r') as f:
        vocab = json.load(f)["vocabulary"]
    input_notes = [68, 67, 25, 78, 35]
    melody = generate_melody(input_notes, vocab, model, to_generate= 100)
    create_midi(melody, output_file= args.output_path)
