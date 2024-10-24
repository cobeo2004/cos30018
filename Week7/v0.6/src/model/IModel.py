from abc import ABC, abstractmethod
import numpy as np

class IModel(ABC):
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x_test: np.ndarray) -> None:
        pass

    @abstractmethod
    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        pass
