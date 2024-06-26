import os
import json
import base64
import struct
import argparse
import numpy as np
import tensorflow as tf
from tqdm import *
from dataset import MelodyDataset

def parse_args():
    parser = argparse.ArgumentParser("Entry script to launch training")
    parser.add_argument("--data-dir", type=str, default = "./data", help="Path to the data directory")
    parser.add_argument("--output-path", type=str, default = './model.json', help="Path to the output file")
    parser.add_argument("--config-path", type=str, required = True, help="Path to the output file")
    parser.add_argument("--checkpoint-path", type = str, default = None,  help="Path to the checkpoint file")
    return parser.parse_args()

def create_model(config, model_path = None):
    rnn_units = config["rnn_units"]
    n_vocab = config["n_vocab"]
    sequence_length = config["seq_length"]
    embedding_dim = config["embedding_dim"]

    if model_path is not None:
        model = tf.keras.models.load_model(model_path)
        return model

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(sequence_length, )),
        tf.keras.layers.Embedding(n_vocab, embedding_dim),
        tf.keras.layers.LSTM(units = rnn_units),
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
                "batch_input_shape": cfg["batch_input_shape"] if "batch_input_shape" in cfg else cfg["batch_shape"],
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
    output_path = args.output_path
    ckpt = args.checkpoint_path
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset = MelodyDataset({"data_dir": data_dir, "sequence_length": config["seq_length"]})
    X, y = dataset.get_training_data()
    vocabulary = dataset.get_vocab()
    config["n_vocab"] = len(vocabulary)
    model = create_model(config, ckpt)
    model.summary()
    model.fit(X, y, epochs=config["epoch_num"], batch_size = config["batch_size"])
    get_model_for_export(output_path, model, vocabulary)

if __name__ == "__main__":
    main()