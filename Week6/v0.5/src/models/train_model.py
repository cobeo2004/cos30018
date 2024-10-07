from typing_extensions import Tuple, List
import numpy as np
from .conf import ModelConfig
from .make_dynamic_model import make_dynamic_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from src.processing import ENTRY_POINT


def train_model(
    # List of model configuration dictionaries
    model_config: List[ModelConfig],
    shape: Tuple[int, int],  # Shape of the input data (n_steps, n_features)
    x_train: np.ndarray,  # Training input data
    y_train: np.ndarray,  # Training target data
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    early_stopping_restore_best_weights: bool = True,
    loss: str = 'mse',
    optimizer: str = 'adam',
    metrics: List[str] = ['accuracy']
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train a dynamic model based on the given configuration and data.

    Args:
        model_config: List of dictionaries defining the model architecture.
        shape: Tuple representing the shape of the input data.
        x_train: Input training data.
        y_train: Target training data.
        epochs: Number of epochs to train the model.
        batch_size: Batch size for training.
        validation_split: Fraction of the training data to be used as validation data.
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped.
        early_stopping_restore_best_weights: Whether to restore the model weights from the epoch with the best validation loss.
        loss: Loss function to be used during training.
        optimizer: Optimizer to be used during training.
        metrics: List of metrics to be evaluated by the model during training and testing.

    Returns:
        A tuple containing the trained model and its training history.
    """
    model = make_dynamic_model(shape, model_config)
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    early_cb = EarlyStopping(
        monitor='val_loss', patience=early_stopping_patience, restore_best_weights=early_stopping_restore_best_weights)
    model_train_history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[
                                    early_cb], shuffle=True, validation_split=validation_split)
    return model, model_train_history


def save_model(model: tf.keras.Model, path: str = ENTRY_POINT + "model.keras") -> None:
    model.save(path)
