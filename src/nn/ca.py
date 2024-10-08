from functools import partial
from typing import Callable, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Float


#----------------------------------- Alive Functions ------------------------------------

class IdentityAlive(eqx.Module):
    def __call__(self, node_states: Float[Array, "C H W"]):
        H, W = node_states.shape[1:]
        return jnp.ones_like(node_states, shape=(1, H, W), dtype=jnp.bool_)


class MaxPoolAlive(eqx.Module):
    alive_bit: int
    alive_threshold: float
    max_pool: nn.MaxPool2d

    def __init__(self, alive_threshold, alive_bit, *, key=None):  # key is keyword only
        super().__init__()
        self.alive_bit = alive_bit
        self.alive_threshold = alive_threshold
        self.max_pool = nn.MaxPool2d(3, 1, 1)

    def __call__(self, node_states: Float[Array, "C H W"]):
        pooling = self.max_pool(node_states[self.alive_bit:self.alive_bit + 1])
        return pooling > self.alive_threshold

#----------------------------------- Message Passing ------------------------------------

_conv2d = partial(jax.scipy.signal.convolve2d, mode='same')
_sobel_kernel_x = jnp.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ]) / 8.0
_sobel_kernel_y = _sobel_kernel_x.T


class IdentityAndSobelFilter(eqx.Module):
    kernel_size: Tuple[int, int]

    def __init__(self, kernel_size: int = 3, *, key=None):  # key is keyword only
        super().__init__()
        if kernel_size != 3:
            raise NotImplementedError
        self.kernel_size = kernel_size, kernel_size

    def __call__(self, inputs: Float[Array, "C H W"]):
        x_conv = jax.vmap(_conv2d, in_axes=(0, None))(inputs, _sobel_kernel_x)
        y_conv = jax.vmap(_conv2d, in_axes=(0, None))(inputs, _sobel_kernel_y)
        return jnp.concatenate([inputs, x_conv, y_conv], axis=0)


#------------------------------------ Dummy layers --------------------------------------

class ConstantContextEncoder(eqx.Module):
    state_size: int
    grid_size: Tuple[int, int]

    def __call__(self, *_):
        return np.zeros((1, self.state_size))

    @property
    def input_shape(self):
        return (self.state_size,)


class IdentityControlFn(eqx.Module):
    def __call__(self, cell_state, input_embedding, *, key=None):
        # return jnp.repeat(input_embedding, 3, 0)
        dummy_weights = np.zeros((len(input_embedding), *cell_state.shape[1:]))
        dummy_ctrl = jnp.zeros_like(cell_state)
        return  dummy_ctrl, dummy_weights

#----------------------------------- Output select --------------------------------------

class SliceOutput(eqx.Module):
    dim: int
    start_idx: int
    end_idx: int
    squashing_function: Callable

    def __init__(self, dim, end_idx, start_idx=0, squashing_function=lambda x: x, *, key=None):
        super().__init__()

        self.dim = dim
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.squashing_function = squashing_function

    def __call__(self, inputs, *, key=None):
        outputs = jax.lax.slice_in_dim(inputs, self.start_idx, self.end_idx + 1, 1, self.dim)
        return self.squashing_function(outputs)
