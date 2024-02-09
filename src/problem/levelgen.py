from functools import partial
from collections import OrderedDict
from typing import Optional, OrderedDict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from src.problem.base import QDProblem, Fitness, Descriptor, ExtraScores
from .map_utils import batched_n_islands, batched_lsp, compute_simmetry


class SimpleLevelGeneration(QDProblem):
    height: int
    width: int

    def __init__(
        self,
        height: int,
        width: Optional[int] = None,
    ) -> None:
        if width is None:
            width = height

        self.height = height
        self.width = width

    @property
    def map_shape(self):
        return (self.height, self.width)

    @property
    def descriptor_info(self) -> OrderedDict[str, Tuple[float, ...]]:
        return OrderedDict({
            'path length': (0, self.height * self.width / 2 + self.width),
            'symmetry': (0, self.height * self.width),
        })

    @property
    def score_name(self):
        return "number of connected components"

    @property
    def score_offset(self):
        # Maximum number of connected components is H * W / 2 when there is a checkerboard pattern.
        # To make it clear that maps with no paths are really bad, we will set the score of an
        # empty map to - H * W and the offset to the opposit evalue. Such maps get a score of 0
        return self.height * self.width

    @partial(jax.jit, static_argnums=(0,))
    def score(self, inputs: Float[Array, "H W"]) -> Fitness:
        """
        Computes the validity of a level by assigning a value to how well it satisfies a given set
        of constraints. This is a continuos value to allow the model to explore outside the space
        of satisfying solutions. Notice that there is no ``optimal'' level, just levels that do or
        do not satisfy the constraints, though any given level can certainly be further away from
        satisfying said constraint than other levels.
        """
        # TODO: rewrite score and measures to only compute the adjacency matrix once
        n_connected_components = jax.pure_callback(
            batched_n_islands,
            jnp.empty((1,)),
            inputs,
            vectorized=True,
        ).squeeze()

        # add max number of connected components to ensure quality scores are positive
        return -n_connected_components + self.score_offset

    @partial(jax.jit, static_argnums=(0,))
    def compute_measures(self, inputs: Float[Array, "H W"]) -> Descriptor:
        path_length = jax.pure_callback(
            batched_lsp,
            jnp.empty((1,)),
            inputs,
            vectorized=True,  # this will sync across all vmaps up to this point
        )

        symmetry = compute_simmetry(inputs, (self.height, self.width))

        return jnp.concatenate([path_length, symmetry])

    @partial(jax.jit, static_argnums=(0,))
    def extra_scores(self, _) -> ExtraScores:
        return {"dummy": jnp.empty(0)}

    def random_scores(self, n_samples, key):
        scores = jnp.zeros(n_samples, dtype=jnp.float32)

        measure_info = self.descriptor_info
        k1, k2 = jax.random.split(key)

        path_length = jax.random.uniform(k1, shape=(n_samples,)) * measure_info['path length'][1]
        symmetry = jax.random.uniform(k2, shape=(n_samples,)) * measure_info['symmetry'][1]

        return scores, jnp.stack([path_length, symmetry], axis=1), {"dummy": jnp.empty(0)}

    def __call__(self, inputs):
        assert inputs.shape[-2:] == (self.map_shape)
        return super().__call__(inputs)
