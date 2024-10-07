import matplotlib.pyplot as plt
import tensorflow as tf


def metric_plot(model_train_history: tf.keras.callbacks.History, metric_1: str, metric_2: str, plot_name: str):
    metric_1_values = model_train_history.history[metric_1]
    metric_2_values = model_train_history.history[metric_2]
    epochs = range(len(metric_1_values))
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metric_1_values, label=metric_1)
    plt.plot(epochs, metric_2_values, label=metric_2)
    plt.title(f'{plot_name} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
