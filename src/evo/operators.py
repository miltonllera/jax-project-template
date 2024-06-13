import jax
import jax.lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array


def _discrete_mutation(
    x: Float[Array, "S"],
    mutation_prob: float,
    min_val: float,
    max_val: float,
    key: jax.Array,
):
    k1, k2 = jr.split(key)

    values = jnp.arange(min_val, max_val + 1, 1.0)
    to_mutate = jr.bernoulli(k1, mutation_prob, (len(x),)).astype(jnp.float32)
    mutations = jr.choice(k2, values, (len(x),))

    return x * (1 - to_mutate) + mutations * to_mutate


def discrete_mutation(
    x: Float[Array, "N S"],
    key: jax.Array,
    mutation_prob: float,
    min_val: float,
    max_val: float,
):
    keys = jr.split(key, len(x) + 1)
    mutations = jax.vmap(_discrete_mutation, in_axes=(0, None, None, None, 0))(
        x, mutation_prob, min_val, max_val, keys[:-1]
    )

    return mutations, keys[-1]
