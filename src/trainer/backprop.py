from functools import partial
from typing import List, Optional

import jax
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import optax

from src.trainer.base import Trainer
from src.trainer.logger import Logger
from src.trainer.callback import Callback, MonitorCheckpoint
from src.model.base import FunctionalModel
from src.task.base import Task


class BackpropTrainer(Trainer):
    def __init__(
        self,
        task: Task,
        optim: optax.GradientTransformation,
        steps: int = 1000,
        val_steps: int = 100,
        val_freq: Optional[int] = None,
        grad_accum: int = 1,
        logger: Optional[List[Logger]] = None,
        callbacks: Optional[List[Callback]] = None,
        use_progress_bar: bool = True,
    ):
        super().__init__(
            task, steps, val_steps, val_freq, logger, callbacks, use_progress_bar
        )

        if grad_accum > 1:
            optim = optax.MultiSteps(optim, every_k_schedule=grad_accum)

        self.optim = optim

    def run(
        self,
        model: FunctionalModel,
        key: jax.Array
    ):
        train_key, _ = jr.split(key)
        params, statics = model.partition()

        def eval_fn(params, task_state, key):
            m = statics.instantiate(params)
            return self.task.eval(m, task_state, key=key)

        def grad_step(train_state, task_state, key):
            params, opt_state = train_state

            grad_eval = eqx.filter_value_and_grad(eval_fn, has_aux=True)
            (_, log_dict), grads = grad_eval(params, task_state, key)

            updates, opt_state = self.optim.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)

            return (params, opt_state), log_dict

        def validation_fn(params, task_state, key):
            m = eqx.combine(params, statics)
            return self.task.validate(m, task_state, key)

        def val_step(params, task_state, key):
            key, val_key = jr.split(key)
            results = validation_fn(params, task_state, val_key)
            return (params, key), results

        return self._fit_loop(model, params, grad_step, val_step, key=train_key)

    def init_train(self, params, key: jax.Array):
        opt_state = self.optim.init(params)
        task_state = self.task.init('train', key)
        return (params, opt_state), task_state

    def init_val_from_train(self, train_state, key):
        task_state = self.task.init('val', key)
        return train_state[0], task_state

    def format_log_dict(self, stage, log_dict):
        return jtu.tree_map(lambda x: x.item(), super().format_log_dict(stage, log_dict))

    def get_best_model(self, state):
        # look for the es_state in the callbacks, if not use the last one
        if self.callbacks is not None:
            ckpt = [c for c in self.callbacks if isinstance(c, MonitorCheckpoint)]
            if len(ckpt) > 0:
                state = ckpt[0].best_state
        return state[0]
