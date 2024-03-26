import numpy as np
import argparse
import tensorflow as tf
import os
import json
from music21 import instrument, note, stream
import pandas as pd
from keract import get_activations

np.set_printoptions(threshold=np.inf)

# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.
MELODY_SIZE = 130

SEQUENCE_LENGTH = 10

def write_to_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)

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
    parser.add_argument('--config-path', type=str, default='config.json', help='Path to the config file')
    return parser.parse_args()


def load_model_from_ckpt(ckpt):
    return tf.keras.models.load_model(ckpt)

def load_stateful_model(config, model):
    rnn_units = config["rnn_units"]
    n_vocab = config["n_vocab"]
    sequence_length = config["seq_length"]

    stateful_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length, 1)),
        tf.keras.layers.LSTM(units = rnn_units),
        tf.keras.layers.Dense(n_vocab)
    ])
    stateful_model.compile(loss= tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    stateful_model.set_weights(model.get_weights()) 
    return stateful_model


def generate_melody(input_notes, vocab, model, seq_length = SEQUENCE_LENGTH, to_generate = 10):
    input_notes = input_notes[-seq_length: ]
    for i in range(seq_length - len(input_notes)):
        input_notes.insert(0, np.random.choice(vocab))
    prediction_output = []

    temperature = 1.0
    for i in range(to_generate):
        prediction_input = np.array(input_notes).reshape(1, seq_length, 1) / MELODY_SIZE

        # activations = get_activations(model, prediction_input, auto_compile=True)
        # [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]
        # print(activations['lstm_1'])

        prediction_logits = model.predict(prediction_input)

        print(prediction_logits)

        prediction_logits = prediction_logits / temperature

        predicted_ids = tf.random.categorical(prediction_logits, num_samples=1)
        prediction = tf.squeeze(predicted_ids, axis=-1)

        prediction_probs = tf.keras.activations.softmax(tf.convert_to_tensor(prediction_logits))
        print(prediction[0], prediction_probs[0][prediction[0]])

        prediction_output.append(vocab[prediction[0]])
        input_notes = input_notes[1:] + [vocab[prediction[0]]]

    return prediction_output
    
def generate_melody_stateful(input_notes, vocab, model, seq_length = SEQUENCE_LENGTH, to_generate = 10):
    input_notes = input_notes[-seq_length: ]
    for i in range(seq_length - len(input_notes)):
        input_notes.insert(0, np.random.choice(vocab))
    prediction_output = []

    temperature = 1.0
    for i in range(to_generate):
        print(input_notes)
        prediction_input = np.array(input_notes).reshape(1, len(input_notes))
        prediction_logits = model.predict(prediction_input)

        # print(prediction_logits)

        prediction_logits = prediction_logits / temperature

        predicted_ids = tf.random.categorical(prediction_logits, num_samples=1)
        prediction = tf.squeeze(predicted_ids, axis=-1)
        # print(prediction)

        prediction_output.append(vocab[prediction[0]])
        input_notes = [vocab[prediction[0]]]

    return prediction_output

def export_weights(model):
    weight_np = model.get_weights()
    
    for idx, layer in enumerate(weight_np):
        print(layer.shape)
        write_to_file(f"model_weight_{idx:02}.txt", str(layer))
        # for weight_group in layer:
        #     flatten = weight_group.reshape(-1).tolist()
        #     # print("flattened length: ", len(flatten))
        #     for i in flatten:
        #         weight_bytes.extend(struct.pack("@f", float(i)))

def create_midi(prediction_output, output_file='test_output.mid'):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    # offset = 0
    # output_notes = []

    # # create note and chord objects based on the values generated by the model
    # for pattern in prediction_output:
    #     new_note = note.Note(pattern)
    #     new_note.offset = offset
    #     new_note.storedInstrument = instrument.Piano()
    #     output_notes.append(new_note)
    #     # increase offset each iteration so that notes do not stack
    #     offset += 0.5

    # midi_stream = stream.Stream(output_notes)
    midi_stream = noteArrayToStream(prediction_output)
    midi_stream.write('midi', fp=output_file)

if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = json.load(f)
    checkpoint_path = os.path.join(model_dir, 'model.h5')
    model_config_path = os.path.join(model_dir, 'model.json')
    model = load_model_from_ckpt(checkpoint_path)

    with open(model_config_path, 'r') as f:
        vocab = json.load(f)["vocabulary"]
    config["n_vocab"] = len(vocab)

    stateful_model = load_stateful_model(config, model)
    stateful_model.summary()

    export_weights(model)

    # input_notes = [68]
    input_notes = [68, 129, 30, 31, 32, 35, 129, 37, 35, 33]
    # input_notes = [68, 67, 25, 78, 35, 68, 67, 25, 78, 35, 68, 67, 25, 78, 35, 68, 67, 25, 78, 35, 68, 67, 25, 78, 35, 68, 67, 25, 78, 35, 68, 67, 25, 78, 35, 68, 67, 25, 78, 35]
    # melody = generate_melody_stateful(input_notes, vocab, stateful_model, to_generate= 100)
    melody = generate_melody(input_notes, vocab, model, seq_length=len(input_notes), to_generate=1)
    create_midi(melody, output_file= args.output_path)
