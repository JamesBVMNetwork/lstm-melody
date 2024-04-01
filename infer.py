import random
import os
import json
import tensorflow as tf
import argparse
import numpy as np
from music21 import note, chord, instrument, stream


SEQUENCE_LENGTH = 20

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using a trained model')
    parser.add_argument('--model-dir', type=str, default='model', help='Directory to load model')
    parser.add_argument('--output-path', type=str, default='output.mid', help='Path to save generated music')
    return parser.parse_args()


def load_model_from_ckpt(ckpt):
    return tf.keras.models.load_model(ckpt)



def generate_melody(input_notes, vocab, model, seq_length = SEQUENCE_LENGTH, to_generate = 10):
    selected_notes = []
    for note in input_notes:
        if note in vocab:
            selected_notes.append(note)
    selected_notes = selected_notes[-seq_length:]
    for i in range(seq_length - len(selected_notes)):
        selected_notes.append(random.randint(0, len(vocab) - 1))
    prediction_output = []
    temperature = 1.0
    for i in range(to_generate):
        prediction_input = np.array(selected_notes).reshape(1, seq_length)
        prediction_logits = model.predict(prediction_input)

        prediction_logits = prediction_logits / temperature

        predicted_ids = tf.random.categorical(prediction_logits, num_samples=1)
        prediction = tf.squeeze(predicted_ids, axis=-1)

        prediction_output.append(vocab[prediction[0]])
        selected_notes = selected_notes[1:] + [prediction[0]]

    return prediction_output

def handle_output_notes(prediction_output):
    output_notes = []
    offset = 0
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5
    
def create_midi(prediction_output, output_file='test_output.mid'):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    
    postprocessed_notes = handle_output_notes(prediction_output)
    midi_stream = stream.Stream(postprocessed_notes)
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
    input_notes = []
    melody = generate_melody(input_notes, vocab, model, to_generate= 100)
    create_midi(melody, output_file= args.output_path)
