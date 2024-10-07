from typing_extensions import List
from .ModelConfig import ModelConfig


# model_config: List[ModelConfig] = [
#     {
#         'type': 'LSTM',
#         'isBidirectional': False,
#         'units': 120,
#         'return_sequences': True,
#         'dropout': 0.2,
#         'activation': 'tanh'
#     },
#     {
#         'type': 'LSTM',
#         'isBidirectional': False,
#         'units': 100,
#         'return_sequences': True,
#         'dropout': 0.2,
#     },
#     {
#         'type': 'GRU',
#         'isBidirectional': False,
#         'units': 80,
#         'return_sequences': True,
#         'dropout': 0.2,
#     },
#     {
#         'type': 'RNN',
#         'isBidirectional': False,
#         'units': 60,
#         'return_sequences': True,
#         'dropout': 0.2,
#         'activation': 'relu'
#     },
#     {
#         'type': 'LSTM',
#         'isBidirectional': False,
#         'units': 40,
#         'return_sequences': False,
#         'dropout': 0.2,
#     },
# ]

# model_config: List[ModelConfig] = [
#     {
#         'type': 'LSTM',
#         'isBidirectional': False,
#         'units': 50,
#         'return_sequences': True,
#         'dropout': 0.2,
#     },
#     {
#         'type': 'LSTM',
#         'isBidirectional': False,
#         'units': 50,
#         'return_sequences': False,
#         'dropout': 0.2,
#     }
# ]

# model_config: List[ModelConfig] = [
#     {
#         'type': 'GRU',
#         'isBidirectional': False,
#         'units': 100,
#         'return_sequences': True,
#         'dropout': 0.2,
#     },
#     {
#         'type': 'GRU',
#         'isBidirectional': False,
#         'units': 100,
#         'return_sequences': False,
#         'dropout': 0.2,
#     }
# ]


# model_config: List[ModelConfig] = [
#     {
#         'type': 'LSTM',
#         'isBidirectional': False,
#         'units': 100,
#         'return_sequences': True,
#         'dropout': 0.2,
#     },
#     {
#         'type': 'GRU',
#         'isBidirectional': False,
#         'units': 100,
#         'return_sequences': False,
#         'dropout': 0.2,
#     }
# ]

model_lstm_config: List[ModelConfig] = [
    {
        'type': 'LSTM',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'LSTM',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'LSTM',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'LSTM',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': False,
        'dropout': 0.2,
    },
]


model_gru_config: List[ModelConfig] = [
    {
        'type': 'GRU',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'GRU',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'GRU',
        'isBidirectional': True,
        'units': 64,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'GRU',
        'isBidirectional': True,
        'units': 64,
        'return_sequences': False,
        'dropout': 0.2,
    }
]

model_rnn_config: List[ModelConfig] = [
    {
        'type': 'RNN',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'RNN',
        'isBidirectional': True,
        'units': 128,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'RNN',
        'isBidirectional': True,
        'units': 64,
        'return_sequences': True,
        'dropout': 0.2,
    },
    {
        'type': 'RNN',
        'isBidirectional': True,
        'units': 64,
        'return_sequences': False,
        'dropout': 0.2,
    }

]
