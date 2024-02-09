from abc import ABC, abstractmethod


class DataModule(ABC):
    @abstractmethod
    def init(self, stage, key):
        raise NotImplementedError

    @abstractmethod
    def next(self, state):
        raise NotImplementedError
