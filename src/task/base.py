from abc import ABC, abstractmethod


class Task(ABC):
    @property
    @abstractmethod
    def mode(self):
        raise NotImplementedError

    @abstractmethod
    def init(self, stage, key):
        raise NotImplementedError

    def next(self, task_state, key):
        raise NotImplementedError

    @abstractmethod
    def eval(self, model, state, key):
        raise NotImplementedError

    @abstractmethod
    def validate(self, model, state, key):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, state, key):
        raise NotImplementedError
