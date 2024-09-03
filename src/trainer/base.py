from abc import abstractmethod, ABC
from typing import List, Optional
from logging import getLogger

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx

from src.model.base import FunctionalModel
from src.trainer.callback import Callback
from src.trainer.logger import Logger
from src.trainer.utils import aot_compilation
from src.task.base import Task
from src.utils.tree import tree_stack

from tqdm import tqdm

_logger = getLogger(__name__)


class Trainer(ABC):
    def __init__(
        self,
        task: Task,
        steps: int,
        val_steps: int,
        val_freq: Optional[int],
        loggers: Optional[List[Logger]],
        callbacks: Optional[List[Callback]],
        use_progress_bar: bool = True,
    ) -> None:
        if loggers is None:
            loggers = []

        if callbacks is None:
            callbacks = []

        if val_freq is None:
            val_freq = steps + 1

        if val_freq > steps:
            _logger.warn(
                "Validation frequency set to value greater than training steps or None which "
                "means no validation iterations will be executed for this run. If this was not"
                f"intended set val_freq to value smaller than {steps}."
            )

        self.task = task
        self.steps = steps
        self.val_steps = val_steps
        self.val_freq = val_freq
        self.loggers = loggers
        self.callbacks = callbacks
        self.use_progress_bar = use_progress_bar

    @abstractmethod
    def run(self, model: FunctionalModel, key: jax.Array) -> eqx.Module:
        raise NotImplementedError

    @abstractmethod
    def init(self, stage, params, *, key):
        raise NotImplementedError

    def format_log_dict(self, stage, log_dict):
        """
        This method just adds the stage from which the metrics where obtained to the key.
        """
        return {f"{stage}/{k}": v for (k, v) in log_dict.items()}

    def _fit_loop(self, model, params, train_step, val_step, *, key):
        init_key, key = jr.split(key)

        train_step, val_step = self._compile_step_fns(train_step, val_step, params)

        _logger.info("Training is starting...")

        train_state, task_state = self.init('train', params, key=init_key)
        self._run_callbacks('train_start', model, train_state)

        for i in tqdm(range(self.steps)):
            task_key, train_key, key = jr.split(key, 3)

            task_state = self.task.next(task_state, task_key)  # change data, e.g. load new batch
            train_state, log_dict = train_step(train_state, task_state, train_key)

            log_dict = self.format_log_dict('train', log_dict)

            self._run_callbacks('train_iter_end', i + 1, log_dict, train_state)

            if (i + 1) % self.val_freq == 0 or (i + 1) == self.steps:
                key, val_key = jr.split(key)
                val_log_dict = self._val_loop(val_step, model, train_state, val_key)
                self._run_callbacks('validation_end', i + 1, val_log_dict, train_state)

        self._run_callbacks('train_end', self.steps, train_state)
        _logger.info('Training completed.')

        return train_state

    def _val_loop(self, val_step, model, params, key):
        val_state, task_state = self.init('val', params, key=key)
        self._run_callbacks('validation_start', model, val_state)

        accum_metrics = []
        for i in range(self.val_steps):
            task_key, val_key, key = jr.split(key)

            task_state = self.task.next(task_state, task_key)
            val_state, metrics = val_step(val_state, task_state, val_key)

            log_dict = self.format_log_dict('val', metrics)
            accum_metrics.append(metrics)

            self._run_callbacks('validation_iter_end', i + 1, log_dict, val_state)

        # steps in a validation loop are averaged
        accum_metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), tree_stack(accum_metrics))

        return self.format_log_dict("val", accum_metrics)

    def _run_callbacks(self, hook_name: str, *args):
        if self.callbacks is not None:
            for c in self.callbacks:
                getattr(c, hook_name)(*args)

        if self.loggers is not None:
            for l in self.loggers:
                getattr(l, hook_name)(*args)

    def _compile_step_fns(self, train_step, val_step, params):
        """
        Performs ahead-of-time compilation of step functions to prevent recompilations.
        """
        # TODO: Add parameters to compilation if needed

        _logger.info("Compiling step functions...")

        train_state = self.init('train', params, key=jr.key(0))
        train_step = aot_compilation(train_step, (*train_state, jr.key(0)))

        if self.val_steps is None:
            val_step = None
        else:
            dummy_val_state = self.init('val', params, key=jr.key(0))
            val_step = aot_compilation(val_step, (*dummy_val_state, jr.key(0)))

        _logger.info("Done.")

        return train_step, val_step
