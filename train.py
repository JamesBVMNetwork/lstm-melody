from mido import MidiFile, MidiTrack, Message
import mido
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils, to_categorical
from keras.models import load_model
import os
from tqdm import *

import pygame
# import IPython
import matplotlib.pyplot as plt
# import librosa.display
# from IPython import *
from music21 import *
from music21 import converter, instrument, note, chord, stream, midi
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import struct
import base64
import json
import site
import argparse


# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.
VOCAB_SIZE = 130

def parse_args():
    parser = argparse.ArgumentParser("Entry script to launch training")
    parser.add_argument("--data-dir", type=str, default = "./data", help="Path to the data directory")
    parser.add_argument("--output-dir", type=str, default = "./outputs", help = "Path to output directory")
    parser.add_argument("--config-path", type=str, required = True, help="Path to the output file")
    parser.add_argument("--checkpoint-path", type = str, default = None,  help="Path to the checkpoint file")
    return parser.parse_args()

def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(
        np.round(stream.flat.highestTime / 0.25))  # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append([np.round(
                element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(
                element.quarterLength / 0.25), element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame(
        {'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    # sort the dataframe properly
    df = df.sort_values(['pos', 'pitch'], ascending=[True, False])
    df = df.drop_duplicates(subset=['pos'])  # drop duplicate values
    # part 2, convert into a sequence of note events
    # set array full of no events by default.
    output = np.zeros(total_length+1, dtype=np.int16) + \
        np.int16(MELODY_NO_EVENT)
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos == i].empty:
            # pick the highest pitch at each semiquaver
            n = df[df.pos == i].iloc[0]
            output[i] = n.pitch  # set note on
            output[i+n.dur] = MELODY_NOTE_OFF
    return output

def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    # calculate durations and change to quarter note fractions
    df.duration = df.duration.diff(-1) * -1 * 0.25
    df = df.fillna(0.25)
    return df[['code', 'duration']]

def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            # bit of an oversimplification, doesn't produce long notes.
            new_note = note.Rest()
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream

# making data to train
def make_training_data(dirname):
    note_on = []
    for currentpath,_, files in os.walk(dirname):
        for file in files:
            path = os.path.join(currentpath, file)
            if path.endswith('.mid') or path.endswith('.midi'):
                mid = MidiFile(path)
                for j in range(len(mid.tracks)):
                    for i in mid.tracks[j]:
                        if str(type(i)) != "<class 'mido.midifiles.meta.MetaMessage'>":
                            x = str(i).split(' ')
                            if x[0] == 'note_on':
                                note_on.append(int(x[2].split('=')[1]))

    # making data to train
    training_data = []
    labels = []
    for i in range(20, len(note_on)):
        inputs = note_on[i-20:i]
        # inputs = to_categorical(inputs, num_classes=VOCAB_SIZE)
        training_data.append(inputs)
        targets = [note_on[i]]
        targets = to_categorical(targets, num_classes=VOCAB_SIZE)
        labels.append(targets)

    return training_data,labels

def create_model(rnn_units, model_path = None):

    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
        return model

    model = tf.keras.models.Sequential()
    
    model.add(LSTM(rnn_units, input_shape=(20, 1), unroll=True,
                return_sequences=True, implementation=1))
    
    model.add(LSTM(rnn_units, input_shape=(20, 1), unroll=True,
                return_sequences=True, implementation=1))
    
    model.add(Dropout(0.2))
    model.add(Dense(130, 'softmax'))
    model.compile(loss='MSE', optimizer='adam')
    return model

def train(training_data, labels, config, output_dir = './outputs', checkpoint_path = None):
    num_units = config['rnn_units']
    num_epochs = config['epoch_num']
    batch_size = config['batch_size']
    
    early_stop = True

    model = create_model(num_units, checkpoint_path)

    early_stop_cb = EarlyStopping(monitor='val_loss', patience=20)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, 'model.h5'), 
        save_best_only=True, 
        monitor = "loss",
        verbose=1
    )

    training_data = np.array(training_data)
    training_data = training_data.reshape(
        (training_data.shape[0], training_data.shape[1], 1))
    labels = np.array(labels)

    # train
    X_train, X_test, y_train, y_test = train_test_split(
        training_data, labels, test_size=0.05, random_state=42)
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=[checkpoint_callback, early_stop_cb])
    output_checkpoint_file = os.path.join(output_dir, 'model.h5')
    output_file = os.path.join(output_dir, 'model.json')
    model.save(output_checkpoint_file)
    get_model_for_export(output_file, model)

def get_weights(model):
    weights = []
    for layer in model.layers:
        w = layer.get_weights()
        print(layer.name)
        print([g.shape for g in w])
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

def get_model_for_export(fname, model):
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
    with open(fname, "w") as f:
        json.dump({
            "model_name": "musicnetgen",
            "layers_config": compressed_config,
            "weight_b64": weight_base64,
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

    training_data,labels=None,None
    training_data,labels=make_training_data(data_dir)
    

    train(training_data, labels, config, output_dir = output_dir, checkpoint_path = ckpt)

if __name__ == "__main__":
    main()