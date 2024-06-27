import numpy as np
import argparse
import tensorflow as tf
import os
import json
from music21 import note, stream, chord

SEQUENCE_LENGTH = 20

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using a trained model')
    parser.add_argument('--model-dir', type=str, default='model', help='Directory to load model')
    parser.add_argument('--output-path', type=str, default='output.mid', help='Path to save generated music')
    return parser.parse_args()


def load_model_from_ckpt(ckpt):
    return tf.keras.models.load_model(ckpt)

def generate_melody(input_notes, vocab, model, seq_length = SEQUENCE_LENGTH, to_generate = 10):
    input_notes = input_notes[-seq_length: ]
    vocab_indices = [i for i in range(len(vocab))]
    for i in range(seq_length - len(input_notes)):
        input_notes.insert(0, np.random.choice(len(vocab_indices)))
    prediction_output = []
    temperature = 1.0
    for i in range(to_generate):
        prediction_input = np.array(input_notes).reshape(1, seq_length)
        prediction_logits = model.predict(prediction_input)
        prediction_logits = prediction_logits / temperature

        predicted_ids = tf.random.categorical(prediction_logits, num_samples=1)
        prediction = tf.squeeze(predicted_ids, axis=-1)
        prediction_output.append(vocab[prediction[0]])
        input_notes = input_notes[1:] + [prediction[0]]
    
    return prediction_output

def chords_n_notes(Snippet):
    Melody = []
    offset = 0 
    for i in Snippet:
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".")
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)            
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        else: 
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        offset += 1
    Melody_midi = stream.Stream(Melody)   
    return Melody_midi
    
def create_midi(prediction_output, output_file='test_output.mid'):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    
    Melody_midi = chords_n_notes(prediction_output)
    Melody_midi.write('midi', fp=output_file)

if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    checkpoint_path = os.path.join(model_dir, 'model.h5')
    model_config_path = os.path.join(model_dir, 'model.json')
    model = load_model_from_ckpt(checkpoint_path)
    model.summary()
    with open(model_config_path, 'r') as f:
        vocab = json.load(f)["vocabulary"]
    input_notes = [0, 3, 2, 0]
    melody = generate_melody(input_notes, vocab, model, to_generate= 100)
    create_midi(melody, output_file = args.output_path)
