from abc import ABC, abstractmethod
from typing import Any


class IChart(ABC):
    @abstractmethod
    def plot(self, data: Any) -> Any:
        pass
