import jax.numpy as jnp


def mse_recons_error(x, y):
    return jnp.power(x - y, 2).sum()


def gaussian_kl(params):
    mean, logvar = params
    return -0.5 * (1 + logvar - jnp.power(mean, 2) - jnp.exp(logvar))
