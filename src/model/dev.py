from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Iterable, Optional, Tuple, Union
from typing_extensions import Self
from jaxtyping import Array, Float, Int, PyTree


from src.model.base import FunctionalModel
from src.utils import Tensor


State = Tuple[Float[Array, "..."], PyTree, Float[Array, "..."], Int, jax.Array]
# class State(namedtuple):
#     """
#     Describe the state of model computations across a developmental + policy evaluation run.
#     """
#     inputs: Float[Array, "..."]
#     node_states: PyTree
#     dev_steps: Int
#     rng_key: Array


class DevelopmentalModel(FunctionalModel):
    """
    A generic developmental model which takes an input encoding (aka goal, "DNA", etc.) and
    produces an output by growing it over several steps.
    """
    dev_steps: Union[Int, Tuple[int, int]]
    context_encoder: Callable[[Tensor], Tensor]
    output_decoder: Callable[[PyTree, jax.Array], Tensor]
    inference: bool
    output_dev_steps: bool

    def rollout(
        self,
        inputs: Float[Array, "..."],
        key: jax.Array,
        # state: Optional[State] = None,  # use this to intervene on state values during analysis
    )-> Tuple[Tensor, PyTree]:
        if isinstance(self.dev_steps, (tuple, list)):
            max_dev_steps = self.dev_steps[1]
        else:
            max_dev_steps = self.dev_steps

        init_state = self.init(inputs, key)
        # if state is None:
        #     init_state = self.init(inputs, key)
        # else:
        #     init_state = state

        final_state, cell_states = jax.lax.scan(self.step, init_state, jnp.arange(max_dev_steps))

        output = self.output_decoder(final_state[1], key=final_state[-1])  # type: ignore

        return output, cell_states

    def __call__(
        self,
        inputs: Tensor,
        key: jax.Array,
        # state: Optional[State] = None
    ) -> Tuple[Tensor, PyTree]:
        return self.rollout(inputs, key)

    @abstractmethod
    def step(self, state: State, i: Int) -> Tuple[State, Iterable[State]]:
        raise NotImplementedError

    def return_dev_states(self, mode: bool) -> Self:
        return eqx.tree_at(lambda x: x.output_dev_states, self, mode)


#-------------------------------------------- NCA -------------------------------------------------

class NCA(DevelopmentalModel):
    """
    Neural Cellular Automata based on Mordvintsev et al. (2020) which supports using goal-directed
    generation as in Shudhakaran et al. (2022).

    This class assumes a grid like organization where cell states occupy the leading dimension of
    the vectors. This means that we can use convolution operations for the updates themselves and
    any function that rearranges the dimensions internally must reverse this process when returning
    the results back to the NCA class.
    """
    state_size: int
    grid_size: Tuple[int, int]
    alive_fn: Callable
    control_fn: Callable
    message_fn: Callable
    update_fn: Callable
    # hyperparams
    update_prob: float

    def __init__(
        self,
        state_size: int,
        grid_size: Tuple[int, int],
        dev_steps: Union[int, Tuple[int, int]],
        context_encoder: Callable,
        alive_fn: Callable,
        control_fn: Callable,
        message_fn: Callable,
        update_fn: Callable,
        update_prob: float,
        output_decoder: Optional[Callable],
        output_dev_steps: bool = False
    ):
        if isinstance(dev_steps, Iterable):
            dev_steps = tuple(dev_steps)  # type: ignore

        if output_decoder is None:
            output_decoder = lambda x, key: x

        super().__init__(dev_steps, context_encoder, output_decoder, False, output_dev_steps)
        self.state_size = state_size
        self.grid_size = grid_size
        self.alive_fn = alive_fn
        self.control_fn = control_fn
        self.message_fn = message_fn
        self.update_fn = update_fn
        self.update_prob = update_prob

    def init(self, inputs, key):
        H, W = self.grid_size

        key, init_key = jr.split(key)

        # TODO: Random initialization doesn't seem to work. Cells tend to die or their values diverge.
        # init_states = jnp.zeros((self.state_size, H, W))
        # # random initialization of cell
        # seed = (0.5 * jr.normal(init_key, (self.state_size,))).at[3].set(1)
        # init_states = init_states.at[:, H//2, W//2].set(seed)

        dna_seq_length = self.context_encoder.input_shape[0]
        init_states = jnp.zeros((self.state_size, H, W)).at[:, H//2, W//2].set(1.0)
        init_weights = jnp.zeros_like(init_states, shape=(dna_seq_length, H, W))
        n_dev_steps = self.sample_generation_steps(init_key)

        return (self.context_encoder(inputs), init_states, init_weights, n_dev_steps, key)

    def step(self, state: State, i: Int) -> Tuple[State, Tensor]:
        dev_steps = state[-2]

        def _step(state):
            # TODO: Currently not using the previous control weights are not used, maybe we should
            context, old_states, _, dev_steps, key = state

            updt_key, ctx_key, carry_key = jr.split(key, 3)

            pre_alive_mask = self.alive_fn(old_states)

            ctrl_signal, ctrl_weights = self.control_fn(old_states, context, key=ctx_key)
            # ctrl_weights = ctrl_weights * pre_alive_mask  # this is useful for visualization

            message_vectors = self.message_fn(
                old_states + ctrl_signal * pre_alive_mask.astype(jnp.float32)
            )
            updates = self.update_fn(message_vectors)
            new_states = old_states + updates * self.stochastic_update_mask(updt_key)

            alive_mask = (self.alive_fn(new_states) & pre_alive_mask).astype(jnp.float32)
            new_states = new_states * alive_mask

            # We just need the final output for training, so we use this flag to disable the rest
            if self.output_dev_steps:
                outputs = (new_states, ctrl_weights * pre_alive_mask)
            else:
                outputs = None

            return (context, new_states, ctrl_weights, dev_steps, carry_key), outputs

        # NOTE: in this case 'jax.cond' exectutes both branches during evaluation since the
        # functions are not dependent on the input. We could make it short-circuit by passing i
        # to both  branches, however because this code is usually in a vmap, it wouldn't make a
        # difference as it will be translated into a 'jax.select' operation (which executes both
        # branches regardless of value of the condition).
        # see: https://github.com/google/jax/issues/3103, #issuecomment-1716019765
        return jax.lax.cond(
            i < dev_steps,
            _step,
            lambda state: (state, (state[1], state[2]) if self.output_dev_steps else None),
            state,
        )

    def stochastic_update_mask(self, key: jax.Array):
        return jr.bernoulli(key, self.update_prob, self.grid_size)[jnp.newaxis].astype(jnp.float32)

    def sample_generation_steps(self, key: jax.Array):
        if isinstance(self.dev_steps, tuple):
            steps = jax.lax.cond(
                self.inference,
                lambda: self.dev_steps[1],
                lambda: jr.choice(key, jnp.arange(*self.dev_steps)).squeeze(),  # type: ignore
            )
        else:
            steps = self.dev_steps
        return steps
