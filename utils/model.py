from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from keras.layers import Dropout, Dense, Flatten, Input
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.models import model_from_json, load_model
import json
import os
from utils import const

def build_model(numnber_action, frame_input_shape, score_input_shape, learning_rate=0.001):
    # Frame process model
    frame_1_in = Input(shape=frame_input_shape)
    frame_1 = Conv2D(32, (5, 5), padding='same', strides=(3, 3), activation='relu')(frame_1_in)
    # frame_1 = Conv2D(32, (5,5), padding='valid', strides=(5, 5), activation='relu')(frame_1)
    # frame_1 = AveragePooling2D(pool_size=(2,2))(frame_1)
    frame_1 = Conv2D(64, (5, 5),  padding='same', strides=(3, 3), activation='relu')(frame_1)
    # frame_1 = Conv2D(64, (4, 4),  padding='valid', strides=(5, 5), activation='relu')(frame_1)
    # frame_1 = AveragePooling2D(pool_size=(2,2))(frame_1)
    frame_1 = Conv2D(128, (5, 5),  padding='same', strides=(3, 3), activation='relu')(frame_1)
    # frame_1 = Conv2D(128, (3, 3),  padding='valid', strides=(5, 5), activation='relu')(frame_1)
    frame_1 = AveragePooling2D(pool_size=(2,2))(frame_1)
    frame_1_out = Flatten()(frame_1)
    # Score model
    score_in = Input(shape=score_input_shape)
    concatenated_layer = concatenate([frame_1_out, score_in])
    output_model = Dense(512, activation='relu')(concatenated_layer)
    output_model = Dropout(0.2)(output_model)
    output_model = Dense(numnber_action)(output_model)
    model = Model([frame_1_in, score_in], output_model)
    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    model.summary()
    return model

def prepare_model(model_path, weight_path, learning_rate=0.001, force_recreate=False):
    if os.path.exists(model_path) and not force_recreate:
        print("Load old model")
        with open(model_path, 'r') as file:
            loaded_model_json= file.read()
        model = model_from_json(loaded_model_json)
        if os.path.exists(weight_path):
            print("Load old weight")
            # load weights into new model
            model.load_weights(weight_path)
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
    else:
        print("Build new model")
        model = build_model(const.ACTIONS, const.INPUT_SIZE, const.SCORE_INPUT_SIZE, learning_rate=learning_rate)
    return model

def save_json_model(model, model_path, weight_path):
    json_model = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(json_model)
    # serialize weights to HDF5
    model.save_weights(weight_path)
    print("Saved model to disk")
