from functools import partial
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PyTree

from src.trainer.base import Trainer
from src.trainer.callback import Callback, MonitorCheckpoint
from src.trainer.logger import Logger
from src.trainer.utils import shvmap
from src.evo.strategy import InstantiatedStrategy, Strategy
from src.model.base import FunctionalModel
from src.task.base import Task


class EvoTrainer(Trainer):
    """
    A trainer that uses an evolutionary strategy to fit a model to a particular task.
    """
    def __init__(
        self,
        task: Task,
        strategy: Strategy,
        steps: int = 100,
        val_steps: int = 1,
        val_freq: Optional[int] = None,
        logger: Optional[List[Logger]] = None,
        callbacks: Optional[List[Callback]] = None,
        use_progress_bar: bool = True,
        split_key_over_population: bool = True,
    ):
        super().__init__(
            task, steps, val_steps, val_freq, logger, callbacks, use_progress_bar
        )

        self.strategy = strategy
        self.split_key_over_population = split_key_over_population

    @property
    def n_generations(self):
        return self.steps

    def run(
        self,
        model: FunctionalModel,
        key: jax.Array,
    ):
        train_key, _ = jr.split(key)

        params, statics = model.partition()
        strategy = self.strategy.instantiate(params)

        #------------------------------------- Train loop -----------------------------------------

        # NOTE: We need to decorate these closures with eqx.filter_jit unless we are using AOT
        # compilation inside the loops (as is currently the case). It's unclear to me why this
        # is necssary, but at least it's not triggering recompilations at every iteration.
        # @eqx.filter_jit
        vmap_key_axis = 0 if self.split_key_over_population else None
        @partial(shvmap, in_axes=(0, None, vmap_key_axis), out_axes=(0, (None, 0)))
        def eval_fn(params, task_state, key):
            m = statics.instantiate(params)
            return self.task.eval(m, task_state, key)

        def evo_step(carry, _):
            es_state, task_state, key = carry
            key, ask_key, eval_key = jr.split(key, 3)

            params, es_state = self._strategy.ask(ask_key, es_state)

            if self.split_key_over_population:
                eval_key = jr.split(eval_key, self.strategy.args['popsize'])

            fitness, (task_state, log_dict) = eval_fn(params, task_state, eval_key)
            es_state = self._strategy.tell(params, fitness, es_state)

            return (es_state, task_state, key), log_dict

        vmap_key_axis = 0 if self.split_key_over_population else None
        @partial(shvmap, in_axes=(0, None, vmap_key_axis), out_axes=(0, None))
        def validation_fn(params, task_state, key):
            m = statics.instantiate(params)
            return self.task.validate(m, task_state, key)

        # # HACK: Use this to easily run the model on a QD only task in the inner loop
        # @partial(jax.vmap, in_axes=(0, None, None))
        # def validation_fn(params, task_state, key):
        #     m = statics.instantiate(params)
        #     return self.task.validate(m, task_state, key)
        # # HACK: up to here

        def val_step(carry, _):
            es_state, task_state, key = carry
            key, val_key, ask_key = jr.split(key, 3)

            params, _ = strategy.ask(ask_key, es_state)

            if self.split_key_over_population:
                val_key = jr.split(val_key, self.strategy.args['popsize'])

            results, task_state = validation_fn(params, task_state, val_key)
            # task_state = jtu.tree_map(lambda x: x[0], task_state)

            return (es_state, task_state, key), results

        return self._fit_loop(model, strategy, evo_step, val_step, key=train_key)

        #------------------------------------- Test loop ------------------------------------------

        # best_parameters = self.get_best_model(trainer_state, strategy)
        # best_model = eqx.combine(best_parameters, statics)

        # # @eqx.filter_jit
        # def test_fn(task_state, key):
        #     return self.task.validate(best_model, task_state, key)

        # def test_step(carry, _):
        #     task_state, key = carry
        #     key, test_key = jr.split(key, 2)
        #     metrics, task_state = test_fn(task_state, test_key)
        #     return (task_state, key), metrics

        # self._test_loop(best_model, test_step, trainer_state, key=test_key)

    def init_train(self, params, key):
        strat_key, task_key = jr.split(key)

        # NOTE: this is a hack
        strategy = self.strategy.instantiate(params)
        self._strategy = strategy

        es_state = strategy.init(strat_key)
        task_state = self.task.init('train', task_key)

        return es_state, task_state

    def init_val_from_train(self, train_state, key):
        params = self.get_best_model(train_state[0])
        task_state = self.task.init('val', key)
        return params, task_state

    def format_log_dict(self, stage, log_dict):
        # NOTE: Since the base method adds the stage name in front of the names, this one only needs
        # to compute the population statistics of interest: mean, variance, minimum and maximum
        metrics_dict_raw = super().format_log_dict(stage, log_dict)
        metrics_dict = {}
        for k, v in metrics_dict_raw.items():
            metrics_dict.update({
                f"{k}_min": jnp.min(v).item(),
                f"{k}_max": jnp.max(v).item(),
                f"{k}_mean": jnp.mean(v).item(),
                f"{k}_var": jnp.var(v).item(),
            })

        return metrics_dict

    def get_best_model(self, state):
        # look for the es_state in the callbacks, if not use the last one
        if self.callbacks is not None:
            ckpt = [c for c in self.callbacks if isinstance(c, MonitorCheckpoint)]
            if len(ckpt) > 0:
                state = ckpt[0].best_state
        return self._strategy.param_shaper.reshape_single(state[0].best_member)
