from abc import ABC, abstractmethod


class Regressor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass
