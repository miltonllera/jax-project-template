from abc import ABC, abstractmethod
from typing import Iterable, List
from jaxtyping import Float, Array
import matplotlib.pyplot as plt

from .callback import Callback


class Logger(Callback, ABC):
    def __init__(self) -> None:
        self.owner = None

    @abstractmethod
    def log_scalar(self, step, key: str, value: List):
        raise NotImplementedError

    @abstractmethod
    def log_dict(self, step, log_dict, *args):
        raise NotImplementedError

    @abstractmethod
    def save_artifact(self, name, artifact):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


try:
    from tensorboardX import SummaryWriter

    class TensorboardLogger(Logger):
        """
        Wrapper around the TensorBoard SummaryWriter class.
        """
        def __init__(self, log_dir: str) -> None:
            super().__init__()
            self.summary_writer = SummaryWriter(log_dir)

        def train_iter_end(self, step, log_dict, *args):
            self.log_dict(log_dict, step)

        def train_end(self, *_):
            self.finalize()

        def validation_end(self, step, log_dict, *args):
            self.log_dict(log_dict, step)
            self.finalize()

        def test_end(self, step, log_dict, *args):
            self.log_dict(log_dict, step)
            self.finalize()

        def log_scalar(self, step, key, value):
            if not isinstance(value, list):
                value = [value]
                step = [step]

            assert len(value) == len(step)

            for v,s in zip(value, step):
                if isinstance(value, Array):
                    value = value.item()
                self.summary_writer.add_scalar(key, v, s)

        def log_dict(self, step, log_dict, *args):
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

except ImportError:
    pass


try:
    import wandb

    class WandBLogger(Logger):
        def __init__(self,
            name: str,
            project: str,
            notes: str,
            tags: List[str],
            run_folder: str,
            log_artifacts: bool = False,
            verbose: bool=False
        ):
            wandb.require('core')
            self.name = name
            self.project = project
            self.notes = notes
            self.tags = tags
            self.run_folder = run_folder
            self.log_artifacts = log_artifacts
            self.verbose = verbose
            self._run = None

        def log_scalar(self, key: str, value: List, step):  # type: ignore
            pass

        def save_artifact(self, name, artifact):  # type: ignore
            pass

        def log_dict(self, step, log_dict, *args):  # type: ignore
            if self._run is not None:
                self._run.log(log_dict, step)

        def finalize(self, *_):
            if self._run is not None:
                self._run.finish()

        def train_start(self, *_):
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                notes=self.notes,
                tags=self.tags,
            )

        def train_iter_end(self, step, log_dict, *args):  # type: ignore
            self.log_dict(step, log_dict)

        def train_end(self, *_):
            self.finalize()

        def validation_end(self, step, log_dict, *args):  # type: ignore
            self.log_dict(step, log_dict)

        def test_iter_end(self, step, log_dict, *args):  # type: ignore
            self.log_dict(step, log_dict)

        def test_end(self, *_):
            self.finalize()

except ImportError:
    pass
