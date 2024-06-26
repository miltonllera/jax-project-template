from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from qdax.core.map_elites import EmitterState, Emitter, MapElitesRepertoire, MAPElites as BaseME
from qdax.core.emitters.standard_emitters import MixingEmitter as MixingEmitterBase
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter as CMAOptEmitterBase
from qdax.core.emitters.omg_mega_emitter import (
    OMGMEGAEmitter as OMGMEGAEmitterBase,
    OMGMEGAEmitterState
)
# from qdax.core.emitters.cma_mega_emitter import CMAMEGAEmitter as CMAMEGAEmitterBase
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.types import Centroid, Metrics, Genotype, Fitness, Descriptor, ExtraScores, RNGKey
from typing import Callable, Optional, Tuple
from jaxtyping import Array, Float


SCORING_RESULTS = Tuple[Fitness, Descriptor, ExtraScores]
MPE_STATE = Tuple[MapElitesRepertoire, EmitterState, RNGKey]


def _dummy_scoring_fn(_: Genotype, k: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    return jnp.empty((1,)), jnp.empty((1,)), {'dummy': jnp.empty((1,))}, k


class MAPElites(BaseME):
    """
    Wrapper around qdax's MAP-Elites implementation.

    The qdax implementation does not use the ask-tell interface, which means that scoring functions
    cannot be easily customized. This is necessary if we are applying developmental models to the
    genotypes in the repertoire. However, this is easy to fix: we use the '_emit' function from the
    emitter to implement the ask and the 'add' and 'update' functions from the repertoire and the
    emitter, respectively, for the tell. we override the '__init__' to pass a dummy scoring function
    and delegate the task of providing the fitness values to a different component. Finally, we must
    also overwrite the 'init' function (NOTE this is the state init, not the module init) to pass
    the scores directly since the module does not have access to the scoring function.
    """
    def __init__(self,
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics]
    ) -> None:
        super().__init__(_dummy_scoring_fn, emitter, metrics_function)

    # @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        fitness_and_metrics: SCORING_RESULTS,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        fitnesses, descriptors, extra_scores = fitness_and_metrics

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, centroids=centroids, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def ask(self, mpe_state: MPE_STATE) -> Genotype:
        repertoire, emitter_state, key = mpe_state
        return self._emitter.emit(repertoire, emitter_state, key)[0]

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        genotypes: Float[Array, "..."],
        scores: SCORING_RESULTS,
        mpe_state: MPE_STATE,
    ) -> Tuple[MPE_STATE, Metrics]:
        fitnesses, descriptors, extra_scores = scores
        repertoire, emitter_state, key = mpe_state

        # update map
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        _, key = jr.split(key)

        # update quality-diversity metrics with the results from the current map
        metrics = self._metrics_function(repertoire)

        return (repertoire, emitter_state, key), metrics


#------------------------------------------- Emitters ---------------------------------------------

class MixingEmitter(MixingEmitterBase):
    def init(self, init_genotypes: Genotype, centroids: Centroid, random_key: jax.Array):
        return super().init(init_genotypes, random_key)


class CMAOptEmitter(CMAOptEmitterBase):
    """
    Emitter used to implement CMA-based Map-Elites (i.e. CMA-ME).
    """
    def __init__(self,
        batch_size: int,
        genotype_dim: int,
        sigma_g: float,
        num_descriptors: int = 2,
        num_centroids: int = 100,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
        *,
        random_key: jax.Array,
    ):
        # We must create dummy centroids to initialize the qdax emitter. Most parameters do not need
        # to match the ones actually used by a QD algorithm since these centroids are only used to
        # access their number.
        dummy_centroids, _ = compute_cvt_centroids(
            num_descriptors=num_descriptors,
            num_init_cvt_samples=num_centroids,
            num_centroids=num_centroids,
            minval=[0, 0],
            maxval=[100, 100],
            random_key=random_key
        )
        super().__init__(batch_size, genotype_dim, dummy_centroids, sigma_g, min_count, max_count)

    def init(self, init_genotypes: Genotype, centroids: Centroid, random_key: jax.Array):
        return super().init(init_genotypes, random_key)


class OMGMEGAEmitter(OMGMEGAEmitterBase):
    def __init__(
        self,
        batch_size: int,
        sigma_g: float,
        num_descriptors: int = 2,
    ):
        super().__init__(batch_size, sigma_g, num_descriptors, None)  # type: ignore


    def init(
        self, init_genotypes: Genotype, centroids: Centroid, random_key: RNGKey
    ) -> Tuple[OMGMEGAEmitterState, RNGKey]:
        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)

        # add a dimension of size num descriptors + 1
        gradient_genotype = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=-1), repeats=self._num_descriptors + 1, axis=-1
            ),
            first_genotype,
        )

        # create the gradients repertoire
        gradients_repertoire = MapElitesRepertoire.init_default(
            genotype=gradient_genotype, centroids=centroids
        )

        return (
            OMGMEGAEmitterState(gradients_repertoire=gradients_repertoire),
            random_key,
        )

#---------------------------------------- Scoring Function ----------------------------------------

# Functions to compute overall scores for the QD process.
# The scoring functions aggregates values from the scores generated for the QD maps.
# Genotype to phonotype evaluators compute a value associated *for the mapping process itself*.
# They do not evaluate the quality phenotypes as this is already computed by the repertoire.


SCORING_FUNCTION = Callable[[Metrics, MapElitesRepertoire], float]
GENOTYPE_TO_PHENOTYPE_EVALUATOR = Callable[
    [Genotype, Float[Array, "..."], MapElitesRepertoire], float
]


def coverage_only(qd_metrics, repertoire):
    coverage = qd_metrics['coverage']

    if len(coverage.shape):
        coverage = coverage[-1]

    return coverage / 100


def qd_score_x_coverage(qd_metrics, repertoire):
    qd_score = qd_metrics['qd_score']
    coverage = qd_metrics['coverage']
    # coverage is a percetange, normalize it to (0, 1)
    return qd_score * coverage / 100


def genotype_to_phenotype_pairwise_dissimilarity_difference(genotype, phenotypes, repertoire):
    def dissimilarity(arr1, arr2):
        return (arr1[None] != arr2).sum(axis=1) / jnp.size(arr1)

    def iter_dissimilarity(g):
        # g's shape is (pop_size, ...): because we wish to perform paiirwise similarity we
        # apply vmap over the first imput and rely on broadcasting.
        return jax.vmap(dissimilarity, in_axes=(0, None))(g, g)

    # Here both genotype and phenotype have shape (n_iters, pop_size, ...)
    # the first vmap applies the function over iterations
    pairwise_genotype_dissimilarity = jax.vmap(iter_dissimilarity)(genotype)
    pairwise_phenotype_dissimilarity = jax.vmap(iter_dissimilarity)(phenotypes)

    diff = pairwise_genotype_dissimilarity - pairwise_phenotype_dissimilarity
    return -diff.mean(axis=[1, 2]).abs()  # mean over pop_size, pop_size dims

# def genotype_to_phenotype_pairwise_difference(genotype, phenotypes, repertoire):
#     def similarity(arr1, arr2):
#         return (arr1 == arr2).sum() / jnp.size(arr1)

#     def for_iter_sim(g):
#         # g's shape is (pop_size, ...): because we wish to perform paiirwise similarity we
#         # apply vmap over the first imput and rely on broadcasting.
#         return jax.vmap(similarity, in_axes=(0, None))(g, g)

#     # Here both genotype and phenotype have shape (n_iters, pop_size, ...)
#     # the first vmap applies the function over iterations
#     pairwise_genotype_similarity = jax.vmap(for_iter_sim)(genotype)
#     pairwise_phenotype_similarity = jax.vmap(for_iter_sim)(phenotypes)

#     return -(pairwise_genotype_similarity - pairwise_phenotype_similarity).mean()


class QDScoreAggregator:
    def __init__(
        self,
        metric_aggregator: SCORING_FUNCTION = qd_score_x_coverage,
        genotype_to_phenotype_evaluator: GENOTYPE_TO_PHENOTYPE_EVALUATOR = lambda *_: 0.0,
        importance_coefficient: float = 1.0,
    ) -> None:
        self.metric_aggregator = metric_aggregator
        self.genotype_to_phenotype_evaluator = genotype_to_phenotype_evaluator
        self.importance_coefficient = importance_coefficient

    @partial(jax.jit, static_argnames=("self",))
    def apply(
        self,
        genotype_and_phenotype: Tuple[Genotype, Float[Array, "..."]],
        qd_metrics: Metrics,
        repertoire: MapElitesRepertoire
    ):
        qd_score = self.metric_aggregator(qd_metrics, repertoire)
        mapping_score = self.genotype_to_phenotype_evaluator(*genotype_and_phenotype, repertoire)
        return qd_score + self.importance_coefficient * mapping_score

    def __call__(self, *args):
        return self.apply(*args)
