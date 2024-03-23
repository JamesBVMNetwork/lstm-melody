import numpy as np
import argparse
import tensorflow as tf
from music21 import instrument, note, stream, chord


# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.
MELODY_SIZE = 130

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using a trained model')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to the checkpoint file')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output')
    return parser.parse_args()


def load_model_from_ckpt(ckpt):
    return tf.keras.models.load_model(ckpt)



def generate_melody(input_notes, model, seq_length = 100, to_generate = 100):
    input_notes = input_notes[-seq_length: ]
    if len(input_notes) < seq_length:
        input_notes = [MELODY_NO_EVENT for _ in range(seq_length - len(input_notes))] + input_notes
    prediction_output = []
    for i in range(to_generate):
        prediction_input = np.array(input_notes).reshape(1, seq_length)
        prediction = model.predict(prediction_input)
        prediction = np.argmax(prediction, axis=1)
        prediction_output.append(prediction[0])
        input_notes = input_notes[1:] + [prediction[0]]

    return prediction_output

if __name__ == '__main__':
    args = parse_args()
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path
    model = load_model_from_ckpt(checkpoint_path)
    model.summary()
    input_notes = [MELODY_NO_EVENT for _ in range(100)]
    melody = generate_melody(input_notes, model)
    print(melody)