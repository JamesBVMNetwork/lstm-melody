import os
from tqdm import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import struct
import base64
import json
import pickle
import argparse
from music21 import converter, note, chord, instrument


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
    return notes

def parse_args():
    parser = argparse.ArgumentParser("Entry script to launch training")
    parser.add_argument("--data-dir", type=str, default = "./data", help="Path to the data directory")
    parser.add_argument("--output-dir", type=str, default = "./output", help="Path to the output directory")
    parser.add_argument("--config-path", type=str, required = True, help="Path to the output file")
    parser.add_argument("--checkpoint-path", type = str, default = None,  help="Path to the checkpoint file")
    return parser.parse_args()

# making data to train
def make_training_data(data_dir, config):
    notes = []
    sequence_length = config["seq_length"]
    file_paths = []

    def list_files_recursive(directory):
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                list_files_recursive(full_path)
            elif os.path.isfile(full_path):
                file_paths.append(full_path)

    list_files_recursive(data_dir)

    resume_path = config['data_resume_path']

    for file_path in tqdm(file_paths):
        try:
            if file_path.endswith('.mid'):
                note_list = extract_notes_from_midi(file_path)
                notes = notes + note_list
            elif file_path.endswith('.pickle'):
                with open(file_path, 'rb') as f:
                    note_list = pickle.load(f)
                    notes = notes + note_list
        except Exception as error: 
            print("Error reading file", file_path)
            print(error)
    
    with open(resume_path, 'wb') as f:
        pickle.dump(notes, f)

    pitchnames = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_index = dict((note, number) for number, note in enumerate(pitchnames))

    inputs = []
    targets = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        inputs.append(sequence_in)
        # inputs.append([note_to_index[char] for char in sequence_in])
        targets.append([note_to_index[sequence_out]])
    
    # reshape the input into a format compatible with LSTM layers
    inputs = np.reshape(inputs, (len(inputs), sequence_length))
    targets = np.array(targets)
    return inputs, targets, note_to_index



def create_model(config, model_path = None):
    embedding_dim = config["embedding_dim"]
    rnn_units = config["rnn_units"]
    n_vocab = config["n_vocab"]
    sequence_length = config["seq_length"]

    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
        return model

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length,)),
        tf.keras.layers.Embedding(n_vocab, embedding_dim, input_length=sequence_length),
        tf.keras.layers.LSTM(units = rnn_units, return_sequences=True),
        tf.keras.layers.LSTM(units = int(2 * rnn_units)),
        tf.keras.layers.Dense(n_vocab)
    ])
    model.compile(loss= tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return model


def get_weights(model):
    weights = []
    for layer in model.layers:
        w = layer.get_weights()
        weights.append(w)
    return weights

def compressConfig(data):
    layers = []
    for layer in data["config"]["layers"]:
        cfg = layer["config"]
        if layer["class_name"] == "InputLayer":
            layer_config = {
                "batch_input_shape": cfg["batch_input_shape"]
            }
        elif layer["class_name"] == "Rescaling":
            layer_config = {
                "scale": cfg["scale"],
                "offset": cfg["offset"]
            }
        elif layer["class_name"] == "Dense":
            layer_config = {
                "units": cfg["units"],
                "activation": cfg["activation"]
            }
        elif layer["class_name"] == "Conv2D":
            layer_config = {
                "filters": cfg["filters"],
                "kernel_size": cfg["kernel_size"],
                "strides": cfg["strides"],
                "activation": cfg["activation"],
                "padding": cfg["padding"]
            }
        elif layer["class_name"] == "MaxPooling2D":
            layer_config = {
                "pool_size": cfg["pool_size"],
                "strides": cfg["strides"],
                "padding": cfg["padding"]
            }
        elif layer["class_name"] == "Embedding":
            layer_config = {
                "input_dim": cfg["input_dim"],
                "output_dim": cfg["output_dim"]
            }
        elif layer["class_name"] == "SimpleRNN":
            layer_config = {
                "units": cfg["units"],
                "activation": cfg["activation"]
            }
        elif layer["class_name"] == "LSTM":
            layer_config = {
                "units": cfg["units"],
                "activation": cfg["activation"],
                "recurrent_activation": cfg["recurrent_activation"],
            }
        else:
            layer_config = None

        res_layer = {
            "class_name": layer["class_name"],
        }
        if layer_config is not None:
            res_layer["config"] = layer_config
        layers.append(res_layer)

    return {
        "config": {
            "layers": layers
        }
    }

def get_model_for_export(output_path, model, vocabulary):
    weight_np = get_weights(model)

    weight_bytes = bytearray()
    for idx, layer in enumerate(weight_np):
        # print(layer.shape)
        # write_to_file(f"model_weight_{idx:02}.txt", str(layer))
        for weight_group in layer:
            flatten = weight_group.reshape(-1).tolist()
            # print("flattened length: ", len(flatten))
            for i in flatten:
                weight_bytes.extend(struct.pack("@f", float(i)))

    weight_base64 = base64.b64encode(weight_bytes).decode()
    config = json.loads(model.to_json())
    # print("full config: ", config)

    compressed_config = compressConfig(config)
    # write to file
    with open(output_path, "w") as f:
        json.dump({
            "model_name": "musicnetgen",
            "layers_config": compressed_config,
            "weight_b64": weight_base64,
            "vocabulary": vocabulary
        }, f)
    return weight_base64, compressed_config

def main():
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    ckpt = args.checkpoint_path
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config['data_resume_path'] = os.path.join(output_dir, 'data.pickle')
    
    X, y, note_to_index = make_training_data(data_dir, config)

    vocabulary = []
    values = []
    for key, value in note_to_index.items():
        vocabulary.append(str(key))
        values.append(value)

    print("Vocabulary: ", vocabulary)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, "model.h5"),
        save_best_only=True,
        monitor="loss",
        mode="min",
        verbose = 1,
    )

    config["n_vocab"] = len(vocabulary)
    model = create_model(config, ckpt)
    model.summary()
    model.fit(X, y, epochs=config["epoch_num"], batch_size = config["batch_size"], callbacks=[checkpoint_callback])
    get_model_for_export(os.path.join(output_dir, "model.json"), model, vocabulary)

if __name__ == "__main__":
    main()