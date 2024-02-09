import os
import os.path as osp
# from matplotlib.animation import PillowWriter
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
from jaxtyping import PyTree
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from src.trainer.base import Trainer
from .run_utils import select_and_unstack


def plot_2d_repertoire(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
):
    os.makedirs(save_dir, exist_ok=True)

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    repertoires = model_outputs[1][0]  # type: ignore

    bd_names = trainer.task.problem.descriptor_names  # type: ignore
    bd_limits = (
        trainer.task.problem.descriptor_min_val, # type: ignore
        trainer.task.problem.descriptor_max_val, # type: ignore
    )

    if iterations_to_plot is not None:
        iterations_to_plot = [
            i if i >= 0 else (range(trainer.task.n_iters) - i) for i in iterations_to_plot  # type: ignore
        ]
    else:
        iterations_to_plot = list(range(trainer.task.n_iters))  # type: ignore

    repertoires = select_and_unstack([repertoires], iterations_to_plot)[0]

    # This only makes sense if we have a population of models
    # rep_names = ["max", "min", "median"]

    # for i,rep in enumerate(repertoires):
    #     max_idx = rep.fitnesses.argmax()
    #     min_idx = rep.fitnesses.argmin()
    #     median_idx = jnp.argsort(rep.fitnesses)[len(rep.fitnesses)//2]

    #     reps = select_and_unstack([rep], [max_idx, min_idx, median_idx])[0]

    #     for r, n in zip(reps, rep_names):
    #         fig, _ = _plot_2d_repertoire_wrapper(r, bd_limits, bd_names)
    #         fig.savefig(osp.join(save_dir, f"{n}-repertoire-iteratoin_{i}.png"))  # type: ignore

    for i,rep in enumerate(repertoires):
        fig, _ = _plot_2d_repertoire_wrapper(rep, bd_limits, bd_names)
        fig.savefig(  # type: ignore
            osp.join(save_dir, f"repertoire-iteratoin_{i}.png"), bbox_inches='tight', dpi=300
        )
        plt.close(fig)


def _plot_2d_repertoire_wrapper(repertoire, bd_limits, bd_names):
    fig, ax = plt.subplots(figsize=(10, 10))

    _, ax = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=bd_limits[0],
        maxval=bd_limits[1],
        repertoire_descriptors=repertoire.descriptors,
        ax=ax
    )

    ax.set_aspect("auto")

    ax.set_xlabel(bd_names[0])
    ax.set_ylabel(bd_names[1])

    return fig, ax
