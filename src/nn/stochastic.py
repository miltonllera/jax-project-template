import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn


class DiagonalGaussian(eqx.Module):
    size: int
    linear: nn.Linear
    training: bool

    def __init__(self, input_size, latent_size):
        super().__init__()
        self.size = latent_size
        self.linear = nn.Linear(input_size, 2 * latent_size)
        self.training = True

    def __call__(self, inputs, key):
        mu, logvar = jnp.split(self.linear(inputs), 2, axis=-1)
        return self.reparam(mu, logvar, key=key), (mu, logvar)

    def reparam(self, mu, logvar, random_eval=False, *, key):
        if self.training or random_eval:
            std = jnp.exp(0.5 * logvar)
            eps = jr.normal(key, std.shape)
            return mu + std * eps
        return mu

    def sample(self, inputs, n_samples=1, *, key):
        h = self.linear(inputs)
        mu, logvar = jnp.split(h[:, None].repeat(n_samples, 1), 2, axis=-1)
        return self.reparam(mu, logvar, random_eval=True, key=key)

    def extra_repr(self):
        return 'size={}'.format(self.size)
