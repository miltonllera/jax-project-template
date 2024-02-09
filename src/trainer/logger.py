from abc import ABC, abstractmethod
from typing import Iterable, List
from jaxtyping import Float, Array

import wandb
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from .callback import Callback


class Logger(Callback, ABC):
    def __init__(self) -> None:
        self.owner = None

    @abstractmethod
    def log_scalar(self, key: str, value: List, step):
        raise NotImplementedError

    @abstractmethod
    def log_dict(self, dict_to_log, step):
        raise NotImplementedError

    @abstractmethod
    def save_artifact(self, name, artifact):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class TensorboardLogger(Logger):
    """
    Wrapper around the TensorBoard SummaryWriter class.
    """
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.summary_writer = SummaryWriter(log_dir)

    def train_iter_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)

    def train_end(self, *_):
        self.finalize()

    def validation_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)
        self.finalize()

    def test_end(self, iter, log_dict, _):
        self.log_dict(log_dict, iter)
        self.finalize()

    def log_scalar(self, key, value, step):
        if not isinstance(value, list):
            value = [value]
            step = [step]

        assert len(value) == len(step)

        for v,s in zip(value, step):
            if isinstance(value, Array):
                value = value.item()
            self.summary_writer.add_scalar(key, v, s)

    def log_dict(self, dict_to_log, step):
        for k, values in dict_to_log.items():
            if isinstance(values, Iterable):
                for v, s in zip(values, step):
                    self.summary_writer.add_scalar(k, v, s)
            else:
                self.summary_writer.add_scalar(k, values, step)

    def save_artifact(self, name, artifact):
        if isinstance(artifact, plt.Figure):
            self.summary_writer.add_figure(name, artifact)
        elif isinstance(artifact, Float):
            self.summary_writer.add_image(name, artifact)
        else:
            raise ValueError(f"Unrecognized type {type(artifact)} for artifact value")

    def finalize(self):
        self.summary_writer.flush()
        self.summary_writer.close()


class WandBLogger(Logger):
    def __init__(self,
        project: str,
        notes: str,
        tags: List[str],
        run_folder: str,
        log_artifacts: bool = False,
		verbose: bool=False
    ):
        self.project = project
        self.notes = notes
        self.tags = tags
        self.run_folder = run_folder
        self.log_artifacts = log_artifacts
        self.verbose = verbose
        self._run = None

    def log_dict(self, iter, log_dict):
        if self._run is not None:
            self._run.log(log_dict, iter)

    def finalize(self, *_):
        if self._run is not None:
            self._run.finish()

    def train_start(self, *_):
        self._run = wandb.init(
            project=self.project,
            notes=self.notes,
            tags=self.tags,
        )

    def train_iter_end(self, iter, log_dict, _):
        self.log_dict(iter, log_dict)

    def train_end(self, *_):
        self.finalize()

    def validation_end(self, iter, log_dict, _):
        self.log_dict(iter, log_dict)

    def test_iter_end(self, iter, log_dict, _):
        self.log_dict(iter, log_dict)

    def test_end(self, *_):
        self.finalize()
