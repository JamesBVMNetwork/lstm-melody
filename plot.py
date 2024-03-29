import tensorflow as tf
import keras

model = keras.models.load_model('./new_funny_piano_outputs/model.h5')

keras.utils.plot_model(model, to_file="model.png", show_shapes=True)