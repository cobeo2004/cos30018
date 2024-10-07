from typing_extensions import TypedDict, Required, Literal, NotRequired


class ModelConfig(TypedDict):
    type: Required[Literal["LSTM", "GRU", "RNN"]]
    isBidirectional: Required[bool]
    units: Required[int]
    return_sequences: Required[bool]
    dropout: Required[float]
    activation: NotRequired[Literal["tanh", "relu", "sigmoid", "softmax", "linear"]]
