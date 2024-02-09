import os
import os.path as osp
from typing import Iterable, Optional, Union

import jax
import matplotlib as mpl
from matplotlib.animation import PillowWriter
from jaxtyping import PyTree

from src.trainer.base import Trainer
from .viz_utils import generate_gif_from_array
from .run_utils import select_and_unstack


#TODO: Better typing information as it is hard to keep track of the structure of model outputs
def plot_generated_levels(
        model_outputs: PyTree,
        model: PyTree,
        trainer: Trainer,
        save_dir: str,
        iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    ):
    os.makedirs(save_dir, exist_ok=True)

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    nca_states = model_outputs[0][2][0]  # type: ignore
    scores, measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(nca_states) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(nca_states)))

    nca_states, scores, measures = select_and_unstack(
        [nca_states, scores, measures],
        iterations_to_plot,
    )

    for i, (ncs, scrs, msrs) in enumerate(zip(nca_states, scores, measures)):
        best_scr = scrs.argmax()  # is one dimensiona
        best_msrs = msrs.argmax(axis=0)  # is two dimensional, where axis 1 has # bd values

        high_scores = ncs[best_scr]
        best_bd_scores = ncs[best_msrs]

        # TODO: Find a way to avoid having to explicitly call output decoder from an NCA.
        high_score_levels = jax.vmap(model.nca.output_decoder)(high_scores)
        best_bd_score_levels = [jax.vmap(model.nca.output_decoder)(s) for s in best_bd_scores]

        cmap = mpl.colormaps['gray']
        best_score_ani = generate_gif_from_array(high_score_levels, cmap=cmap)
        best_bd_score_ani = [generate_gif_from_array(l, cmap=cmap) for l in best_bd_score_levels]

        save_file = osp.join(save_dir, f"best_score-step_{i}.gif")
        best_score_ani.save(save_file, PillowWriter(fps=1))

        for ani, name in zip(best_bd_score_ani, trainer.task.problem.descriptor_names):  #type: ignore
            save_file = osp.join(save_dir, f"best_{name}-step_{i}.gif")
            ani.save(save_file, PillowWriter(fps=1))
