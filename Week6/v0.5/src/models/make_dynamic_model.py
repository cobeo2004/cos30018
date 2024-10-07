from typing_extensions import List, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer, Input, Activation, Bidirectional, GRU, SimpleRNN
from .ModelConfig import ModelConfig


def make_dynamic_model(input_shape: tuple[int, int], config: List[ModelConfig], output_units: Optional[int] = 1):
    model = Sequential()

    first_layer = config[0]
    first_layer_type = first_layer['type']
    if first_layer['isBidirectional']:
        match first_layer_type:
            case "LSTM":
                model.add(Bidirectional(LSTM(
                    units=first_layer['units'], return_sequences=first_layer['return_sequences']), input_shape=input_shape))
            case "GRU":
                model.add(Bidirectional(GRU(
                    units=first_layer['units'], return_sequences=first_layer['return_sequences']), input_shape=input_shape))
            case "RNN":
                model.add(Bidirectional(SimpleRNN(
                    units=first_layer['units'], return_sequences=first_layer['return_sequences']), input_shape=input_shape))
    else:
        match first_layer_type:
            case "LSTM":
                model.add(LSTM(
                    units=first_layer['units'], return_sequences=first_layer['return_sequences'], input_shape=input_shape))
            case "GRU":
                model.add(GRU(
                    units=first_layer['units'], return_sequences=first_layer['return_sequences'], input_shape=input_shape))
            case "RNN":
                model.add(SimpleRNN(
                    units=first_layer['units'], return_sequences=first_layer['return_sequences'], input_shape=input_shape))

    if 'activation' in first_layer:
        model.add(Activation(first_layer['activation']))

    model.add(Dropout(first_layer['dropout']))

    for layer in config[1:]:
        multi_layer_type = layer['type']
        if layer['isBidirectional']:
            match multi_layer_type:
                case "LSTM":
                    model.add(Bidirectional(LSTM(
                        units=layer['units'], return_sequences=layer['return_sequences']), input_shape=input_shape))
                case "GRU":
                    model.add(Bidirectional(GRU(
                        units=layer['units'], return_sequences=layer['return_sequences']), input_shape=input_shape))
                case "RNN":
                    model.add(Bidirectional(SimpleRNN(
                        units=layer['units'], return_sequences=layer['return_sequences']), input_shape=input_shape))
        else:
            match multi_layer_type:
                case "LSTM":
                    model.add(LSTM(
                        units=layer['units'], return_sequences=layer['return_sequences'], input_shape=input_shape))
                case "GRU":
                    model.add(GRU(
                        units=layer['units'], return_sequences=layer['return_sequences'], input_shape=input_shape))
                case "RNN":
                    model.add(SimpleRNN(
                        units=layer['units'], return_sequences=layer['return_sequences'], input_shape=input_shape))

        if 'activation' in layer:
            print(layer['activation'])
            model.add(Activation(layer['activation']))

        model.add(Dropout(layer['dropout']))

    model.add(Dense(units=output_units))

    return model
