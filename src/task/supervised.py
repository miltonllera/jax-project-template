from typing import Dict, Tuple

import jax
import jax.random as jr
from jaxtyping import PyTree

from src.task.base import Task
from src.model.base import FunctionalModel
from src.utils import Tensor


class SupervisedTask(Task):
    """
    Task that evaluates the performance of a model at predicting targets from inputs as determined
    by the provided loss function.
    """
    def __init__(
        self,
        dataset,
        loss_fn,
        prepare_batch = None,
        is_minimization_task: bool = True
    ) -> None:
        super().__init__()

        if prepare_batch is None:
            # by default, discard model state
            prepare_batch = lambda x, y: (x[0], y)

        self.dataset = dataset
        self.loss_fn = loss_fn
        self.prepare_batch = prepare_batch
        self.is_minimization_task = is_minimization_task

    def init(self, stage, training_state, key):
        dataset_state = self.dataset.init(stage, key)
        return dataset_state

    # we don't seem to need this here if we wrap the step functions in a filtered jit or
    # perform ahead-of-time compilation (which is what the code is doing at the momoent).
    # eqx.filter_jit
    def eval(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray,
    ) -> Tuple[Tensor, Tuple[Dict[str, Tensor], PyTree]]:
        """
        Evaluate model fitness on a single batch.
        """
        (pred, y), state = self.predict(model, state, key)
        pred, y = self.prepare_batch(pred, y)

        loss = jax.vmap(self.loss_fn)(pred, y)
        loss = loss.sum() / len(y)
        return loss, (state, dict(loss=loss))

    # eqx.filter_jit
    def validate(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray,
    ) -> Tuple[Dict[str, Tensor], PyTree]:
        #ODO: Run eeeeics other that the loss function here
        (pred, y), state = self.predict(model, state, key)
        pred, y = self.prepare_batch(pred, y)

        loss = jax.vmap(self.loss_fn)(pred, y)
        loss = loss.sum() / len(y)
        return dict(loss=loss), state

    # eqx.filter_jit
    def predict(
        self,
        model: FunctionalModel,
        state: PyTree,
        key: jr.KeyArray,
    ) -> Tuple[Tuple[Tensor, Tensor], PyTree]:
        state = self.dataset.next(state)
        x, y = state[0]

        batched_keys = jr.split(key, len(x))
        pred = jax.vmap(model)(x, batched_keys)

        return (pred, y), state

    def aggregate_metrics(self, metric_values):
        return metric_values

    @property
    def mode(self):
        return "min" if self.is_minimization_task else "max"

    def __call__(self, *args, **kwds):
        return self.eval(*args, **kwds)
