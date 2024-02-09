import jax.random as jr
from typing import Callable, Tuple
from jaxtyping import PRNGKeyArray, PyTree

from src.utils import Tensor
from src.model.base import FunctionalModel
from src.task.loss import mse_recons_error, gaussian_kl


class AutoEncoder(FunctionalModel):
    encoder: Callable[[Tensor], Tensor]
    latent: Callable[[Tensor, PRNGKeyArray], Tuple[Tensor, PyTree]]
    decoder: Callable[[Tensor], Tensor]

    def __call__(self, inputs: PyTree, key: jr.KeyArray) -> Tuple[Tensor, Tuple[Tensor, PyTree]]:
        h = self.encoder(inputs)
        z, params = self.latent(h, key)
        recons = self.decoder(z)
        return recons, (z, params)

    def sample_latent(self, inputs: PyTree, key: jr.KeyArray) -> Tuple[Tensor, PyTree]:
        h = self.encoder(inputs)
        return self.latent(h, key)


def vae_loss(inputs, targets, beta=1.0, recons_error_fn=mse_recons_error, kl_fn=gaussian_kl):
    recons, (_, params) = inputs
    return recons_error_fn(recons, targets) + beta * kl_fn(params)
