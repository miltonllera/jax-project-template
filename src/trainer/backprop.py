from typing import List, Optional

import jax
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import optax
from typing import Tuple
from jaxtyping import Array, Float, PyTree

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
        opt_state = params, self.optim.init(params)

        #------------------------------------- Train loop -----------------------------------------

        # @partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, (None, 0)))
        def eval_fn(params, task_state, key):
            m = statics.instantiate(params)
            return self.task.eval(m, task_state, key=key)

        def grad_step(carry, _):
            params, opt_state, task_state, key = carry

            key, eval_key = jr.split(key)

            grad_eval = eqx.filter_value_and_grad(eval_fn, has_aux=True)
            (_, (task_state, log_dict)), grads = grad_eval(params, task_state, eval_key)

            updates, opt_state = self.optim.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)

            return (params, opt_state, task_state, key), log_dict

        # @partial(jax.vmap, in_axes=(0, None, None), out_axes=(0, (None, 0)))
        def _validation_fn(params, task_state, key):
            m = eqx.combine(params, statics)
            return self.task.validate(m, task_state, key)

        @eqx.filter_jit
        def val_step(carry, _):
            params, task_state, key = carry
            key, val_key = jr.split(key)
            results, task_state = _validation_fn(params, task_state, val_key)
            return (params, task_state, key), results

        return self._fit_loop(model, opt_state, grad_step, val_step, key=train_key)

        #------------------------------------- Test loop ------------------------------------------

        # @eqx.filter_jit
        # def test_step(carry, _):
        #     model, task_state, key = carry
        #     key, test_key = jr.split(key, 2)
        #     metrics, task_state = self.task.validate(model, task_state, test_key)
        #     return (model, task_state, key), metrics

        # best_parameters = self.get_best_model(trainer_state)
        # best_model = model.set_parameters(best_parameters)

        # self._test_loop(
        #     best_model,
        #     test_step,
        #     trainer_state,
        #     key=test_key,
        # )

    def init(
        self,
        stage: str,
        optimizer: Tuple[Float[Array, "..."], optax.OptState],
        trainer_state: PyTree,
        *,
        key: jax.Array,
    ):
        if stage == "train":
            params, opt_state = optimizer
            task_key, loop_key = jr.split(key)
            task_state = self.task.init("train", None, task_key)
            state = params, opt_state, task_state, loop_key

        elif stage == "val":
            task_key, loop_key = jr.split(key)
            params, _, task_state, _ = trainer_state
            task_state = self.task.init("val", task_state, key)
            state = params, task_state, loop_key

        else:
            #TODO: implement test stage initialisation
            raise ValueError(f"Unrecognized stage {stage}.")

        return state

    def format_log_dict(self, stage, log_dict):
        return jtu.tree_map(lambda x: x.item(), super().format_log_dict(stage, log_dict))

    def get_best_model(self, state):
        # look for the es_state in the callbacks, if not use the last one
        if self.callbacks is not None:
            ckpt = [c for c in self.callbacks if isinstance(c, MonitorCheckpoint)]
            if len(ckpt) > 0:
                state = ckpt[0].best_state
        return state[0]
