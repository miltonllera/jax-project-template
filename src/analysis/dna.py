import os
import os.path as osp
from itertools import product
from functools import partial
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.cross_decomposition import PLSCanonical
from jaxtyping import PyTree
# from src.nn.dna import DNAIndependentSampler

from src.trainer.base import Trainer
from .viz_utils import generate_gif_from_array, strip
from .run_utils import select_and_unstack, batched_eval


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 20


#TODO: Currently it's hard to keep track of the structure of the model's outputs. Also, plotting
# functions should not have to deal with any of this indexing stuff or figure saving, they should
# just get what they need and return the created figure to be handled by an outside class.

def plot_dna_decoding_sequence(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    reduction: str = 'none',
    plot_kws: Dict[str, Any] = {},
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    dnas = model_outputs[0][0]

    if model.dna_generator.return_raw_probabilities:
        dnas = dnas.reshape(*dnas.shape[:2], *dna_shape).argmax(-1, keepdims=True)
    else:  # this is a simple string
        dnas = dnas[..., None]

    dna_weights = model_outputs[0][2][1]  # type: ignore
    scores, measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(dna_weights) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(dna_weights)))

    dnas, dna_weights, scores, measures = select_and_unstack(
        [dnas, dna_weights, scores, measures],
        iterations_to_plot,
    )

    for i, (d, dw, scrs, msrs) in enumerate(zip(dnas, dna_weights, scores, measures)):
        best_scr = scrs.argmax()  # is one dimensiona
        # best_msrs = msrs.argmax(axis=0)  # is two dimensional, where axis 1 has # bd values

        high_score_dna = d[best_scr]
        high_score_weights = dw[best_scr]
        # best_bd_dna = d[best_msrs]
        # best_bd_weights = dw[best_msrs]

        if reduction == 'none':
            plot_fn = viusualize_all_weights
        elif reduction == 'maxchar':
            plot_fn = visualize_kth_attended_character
        else:
            raise RuntimeError

        plot_fn = partial(plot_fn, **plot_kws)

        best_score_ani = plot_fn(high_score_dna, high_score_weights, dna_shape)
        # best_bd_score_ani = [
        #     dna_attention_vizualization(d, w, dna_shape, reduction)
        #     for d, w in zip(best_bd_dna, best_bd_weights)
        # ]

        save_file = osp.join(
            save_dir,
            f"best_score-step_{i}" + (f"-{reduction}"  if reduction != 'none' else '') + ".gif")
        best_score_ani.save(save_file, writer='pillow', fps=1)

        # for ani, name in zip(best_bd_score_ani, trainer.task.problem.descriptor_names):  #type: ignore
        #     save_file = osp.join(save_dir, f"best_{name}-step_{i}-({reduction}).gif")
        #     ani.save(save_file, PillowWriter(fps=1))


def viusualize_all_weights(dna, weight_sequence, dna_shape):
    # weight_sequence has shape (ALPHABET_SIZE, H, W)
    seqlen = weight_sequence.shape[0]
    n_cell_rows = weight_sequence.shape[2]
    n_cell_columns = weight_sequence.shape[1] * weight_sequence.shape[3]  # plot the weights horizontally

    weight_sequence = weight_sequence.transpose(0, 2, 3, 1).reshape(seqlen, n_cell_rows, -1)

    fig, (dna_ax, gif_ax) = plt.subplots(
        1, 2, figsize=(22, 3),
        gridspec_kw=dict(width_ratios=[2, n_cell_columns], wspace=5/n_cell_columns),
    )
    # fig.suptitle(plot_name)

    dna_ax.imshow(dna)
    strip(dna_ax)
    dna_ax.set_yticks(np.arange(dna_shape[0]))
    dna_ax.set_yticklabels(np.arange(dna_shape[0]))
    dna_ax.set_ylabel("DNA sequence")

    gif_ax.set_xticks(np.arange(-.5, n_cell_columns, dna.shape[0]), minor=True)
    gif_ax.set_yticks(np.arange(-.5, n_cell_rows), minor=True)

    # Gridlines based on minor ticks
    gif_ax.grid(which='minor', color='w', linestyle='-', linewidth=1.2)

    # Remove minor ticks
    gif_ax.tick_params(which='minor', bottom=False, left=False)

    ani = generate_gif_from_array(weight_sequence, fig, gif_ax)
    return ani


def visualize_kth_attended_character(dna, weight_sequence, dna_shape, k=0):
    # weight_sequence has shape (SEQLEN, ALPHABET_SIZE, H, W)
    n_cell_rows = weight_sequence.shape[2]
    n_cell_columns = weight_sequence.shape[3]

    # argsort is ascending, so negate values before sorting to get descending
    weight_sequence = (-weight_sequence).argsort(axis=1)[:, k]

    fig, (dna_ax, gif_ax) = plt.subplots(
        1, 2, figsize=(5, 3),
        gridspec_kw=dict(width_ratios=[2, n_cell_columns], wspace=2/n_cell_columns),
    )
    # fig.suptitle(plot_name)

    dna_ax.imshow(dna)

    dna_ax.set_xticks([])
    dna_ax.set_yticks(np.arange(dna_shape[0]))
    dna_ax.set_yticklabels(np.arange(dna_shape[0]))
    dna_ax.set_ylabel("DNA sequence")

    gif_ax.set_xticks(np.arange(-.5, n_cell_columns), minor=True)
    gif_ax.set_yticks(np.arange(-.5, n_cell_rows), minor=True)

    # Gridlines based on minor ticks
    gif_ax.grid(which='minor', color='w', linestyle='-', linewidth=1.2)

    # Remove minor ticks
    gif_ax.tick_params(which='minor', bottom=False, left=False)

    colormap = mpl.colormaps['Accent']

    ani = generate_gif_from_array(
        weight_sequence,
        fig,
        gif_ax,
        vrange=(0, dna_shape[0]),
        cmap=colormap,
        colorbar_labels=range(dna_shape[0])
    )
    return ani


def pairwise_dissimilarity(g):
    def dissimilarity(arr1, arr2):
        return (arr1[None] != arr2).sum(axis=1) / jnp.size(arr1)
    # g's shape is (pop_size, ...): because we wish to perform paiirwise similarity we
    # apply vmap over the first imput and rely on broadcasting.
    return jax.vmap(dissimilarity, in_axes=(0, None))(g, g)


def dna_to_output_dissimilarity(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    plot_kws: Dict[str, Any] = {},
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    all_dnas = model_outputs[0][0]
    all_nca_states = model_outputs[0][2][0]  # type: ignore

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(all_nca_states) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(all_nca_states)))

    all_dnas, all_nca_states = select_and_unstack([all_dnas, all_nca_states], iterations_to_plot)

    for i, (dnas, ncs) in enumerate(zip(all_dnas, all_nca_states)):
        # DNAs have shape (popsize, seqlen, alphabet_size)
        # NCA states have shape (popsize, dev_steps, H, W)
        if input_is_distribution:
            dnas = dnas.reshape((len(dnas), *dna_shape)).argmax(-1)

        lvls = jax.vmap(model.nca.output_decoder)(ncs[:, -1]).reshape(len(dnas), -1)

        dna_sim = pairwise_dissimilarity(dnas)
        lvl_sim = pairwise_dissimilarity(lvls)
        sim_diff = dna_sim - lvl_sim

        fig, (dna_ax, lvl_ax, sim_ax) = plt.subplots(ncols=3, figsize=(12, 3))

        dna_ax.imshow(dna_sim, vmin=0, vmax=1)
        im_lvl = lvl_ax.imshow(lvl_sim, vmin=0, vmax=1)
        im_sim = sim_ax.imshow(sim_diff, vmin=-1, vmax=1)

        # print(jnp.corrcoef(dna_sim.ravel(), lvl_sim.ravel()))

        strip(dna_ax)
        strip(lvl_ax)
        strip(sim_ax)

        dna_ax.set_xlabel("DNA dissimilarity")
        lvl_ax.set_xlabel("level dissimilarity")
        sim_ax.set_xlabel("dissimilarity difference")

        plt.colorbar(im_lvl, ax=lvl_ax, pad=0.01)
        plt.colorbar(im_sim, ax=sim_ax, pad=0.01)

        save_file = osp.join(save_dir, f"correlations-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


def dna_distribution_comparison(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    proj_kwargs: Dict[str, Any] = {},
):

    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    all_dnas = model_outputs[0][0]
    all_scores, all_measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(all_dnas) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(all_dnas)))

    random_dnas = jr.normal(jr.key(0), shape=(all_dnas.shape[1], np.prod(dna_shape)))
    if not model.dna_generator.return_raw_probabilities:
        random_dnas = random_dnas.reshape(-1, *dna_shape).argmax(-1)
    # BUG: this doesn't seem to work for some reason. The samples, even though fixed, are
    # displaced from one plot to the next.
    # un_trained_generator = DNAIndependentSampler(*dna_shape, key=jr.key(1))
    # random_dnas = un_trained_generator(len(all_dnas), key=jr.key(2))
    # random_dnas = random_dnas.reshape(-1, np.prod(dna_shape))

    random_lvls = jax.vmap(model.nca, in_axes=(0, None))(random_dnas, jr.key(3))[0]
    random_dna_scores, random_dna_measures, _ = jax.vmap(trainer.task.problem)(random_lvls)  # type: ignore

    # compute PCA for both the metrics and the embedding space
    # metrics = np.concatenate([
    #     np.concatenate([random_dna_scores, all_scores.reshape(-1)])[..., None],
    #     np.concatenate([random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]),
    #     ], axis=1
    # )
    metrics = np.concatenate(
        [random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]
    )
    score_pca =  PCA(1).fit(metrics)

    embeddings = np.concatenate([random_dnas, all_dnas.reshape(-1, all_dnas.shape[-1])])
    embedding_pca = PCA(2).fit(embeddings)

    # compute limits for the plots
    score_proj = score_pca.transform(metrics)
    vmin, vmax = score_proj.min(), score_proj.max()

    dna_proj = embedding_pca.transform(embeddings).T
    xmin, ymin = dna_proj.min(axis=1) - 1.0
    xmax, ymax = dna_proj.max(axis=1) + 1.0

    # format for zip
    all_dnas, all_scores, all_measures = select_and_unstack(
        [all_dnas, all_scores, all_measures], iterations_to_plot
    )

    for i, (dnas, scores, measures) in enumerate(zip(all_dnas, all_scores, all_measures)):
        # X_proj = TSNE(2, **proj_kwargs).fit_transform(X)
        X_proj = embedding_pca.transform(jnp.concatenate([random_dnas, dnas]))
        # M_proj = score_pca.transform(
        #     np.concatenate([
        #         np.concatenate([random_dna_scores, scores])[..., None],
        #         np.concatenate([random_dna_measures, measures]),
        #     ], axis=1
        # )).squeeze()
        M_proj = score_pca.transform(np.concatenate([random_dna_measures, measures]))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

        x, y = X_proj.T

        x_1, x_2 = np.split(x, 2)
        y_1, y_2 = np.split(y, 2)
        c_1, c_2 = np.split(M_proj, 2)

        ax.scatter(x_1, y_1, c=c_1, label='untrained', marker='o',  vmin=vmin, vmax=vmax)
        im = ax.scatter(x_2, y_2, c=c_2, label='learned', marker='x', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, pad=0.01)
        ax.legend()  # type: ignore

        save_file = osp.join(save_dir, f"dna_distribution-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


def dna_measure_projection(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    iterations_to_plot: Optional[Union[int, Iterable[int]]] = None,
    proj_kwargs: Dict[str, Any] = {},
):

    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    all_dnas = model_outputs[0][0]
    all_scores, all_measures = model_outputs[2][0][:2]  #type: ignore

    if iterations_to_plot is not None and not isinstance(iterations_to_plot, Iterable):
        iterations_to_plot = [iterations_to_plot]

    if iterations_to_plot is not None:
        iterations_to_plot = [i if i >= 0 else (len(all_dnas) - i) for i in iterations_to_plot]
    else:
        iterations_to_plot = list(range(len(all_dnas)))

    random_dnas = jr.normal(jr.key(0), shape=(all_dnas.shape[1], np.prod(dna_shape)))
    if not model.dna_generator.return_raw_probabilities:
        random_dnas = random_dnas.reshape(-1, *dna_shape).argmax(-1)
    # BUG: this doesn't seem to work for some reason. The samples, even though fixed, are
    # displaced from one plot to the next.
    # un_trained_generator = DNAIndependentSampler(*dna_shape, key=jr.key(1))
    # random_dnas = un_trained_generator(len(all_dnas), key=jr.key(2))
    # random_dnas = random_dnas.reshape(-1, np.prod(dna_shape))

    # TODO: this is a bit expensive to run just for debugging, so just use dummy scores for now
    random_lvls = jax.vmap(model.nca, in_axes=(0, None))(random_dnas, jr.key(3))[0]
    _, random_dna_measures, _ = jax.vmap(trainer.task.problem)(random_lvls)  # type: ignore

    # random_dna_scores = jr.normal(jr.key(0), shape=(all_dnas.shape[1]))

    # compute PCA for both the metrics and the embedding space
    # metrics = np.concatenate([
    #     np.concatenate([random_dna_scores, all_scores.reshape(-1)])[..., None],
    #     np.concatenate([random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]),
    #     ], axis=1
    # )
    # metrics = np.concatenate(
    #     [random_dna_measures, all_measures.reshape(-1, all_measures.shape[-1])]
    # )
    # score_pca =  PCA(1).fit(metrics)

    embeddings = np.concatenate([random_dnas, all_dnas.reshape(-1, all_dnas.shape[-1])])
    embedding_pca = PCA(1).fit(embeddings)

    # print(embedding_pca.components_)
    # print(embedding_pca.explained_variance_)
    # print(embedding_pca.explained_variance_ratio_)
    # exit()

    # compute limits for the plots
    dna_proj = embedding_pca.transform(embeddings)
    vmin, vmax = dna_proj.min(), dna_proj.max()

    # format for zip
    all_dnas, all_scores, all_measures = select_and_unstack(
        [all_dnas, all_scores, all_measures], iterations_to_plot
    )

    bd_min = trainer.task.problem.descriptor_min_val  # type: ignore
    bd_max = trainer.task.problem.descriptor_max_val  # type: ignore
    bd_names = trainer.task.problem.descriptor_names  # type: ignore

    for i, (dnas, measures) in enumerate(zip(all_dnas, all_measures)):
        # X_proj = TSNE(2, **proj_kwargs).fit_transform(X)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim((bd_min[0], bd_max[0]))  # type: ignore
        ax.set_ylim((bd_min[1], bd_max[1]))  # type: ignore
        ax.set_xlabel(bd_names[0])  # type: ignore
        ax.set_ylabel(bd_names[1])  # type: ignore

        X_proj = embedding_pca.transform(jnp.concatenate([random_dnas, dnas]))
        x, y = np.concatenate([random_dna_measures, measures]).T

        x_1, x_2 = np.split(x, 2)
        y_1, y_2 = np.split(y, 2)
        c_1, c_2 = np.split(X_proj, 2)

        ax.scatter(x_1, y_1, c=c_1, label='untrained', marker='o',  vmin=vmin, vmax=vmax) # type: ignore
        im = ax.scatter(x_2, y_2, c=c_2, label='learned', marker='x', vmin=vmin, vmax=vmax)  # type: ignore
        plt.colorbar(im, ax=ax, pad=0.01)
        ax.legend()  # type: ignore

        save_file = osp.join(save_dir, f"dna_metric_projection-step_{i}.png")
        fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
        plt.close(fig)


def log_dna_guided_nca_metrics(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    (dnas, outputs, _), _, (scores, metrics), ((repertoire, _, _), _) = model_outputs

    fitness, extra_terms = trainer.task.overall_fitness(  # type: ignore
        (dnas, outputs), metrics, repertoire, jr.key(0)
    )
    metrics['fitness'] = fitness

    bd_bests = scores[1].max(axis=1)
    bd_means = scores[1].mean(axis=1)
    stat_vals = jnp.concatenate([bd_bests, bd_means], axis=1).T

    dict_keys = product(("max", "mean"), trainer.task.problem.descriptor_names)  # type: ignore

    bd_metrics = {f"{stat}_{bd}": v for ((stat, bd), v) in zip(dict_keys, stat_vals)}
    metrics =  {**metrics, **extra_terms, **bd_metrics}

    np.savez(osp.join(save_dir, "metrics.npz"), **metrics)


def dna_to_metric_sparse_regression(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    dnas = model_outputs[0][0]
    scores, measures = model_outputs[2][0][:2]  #type: ignore

    # set up the data
    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    if input_is_distribution:
        flattened_dnas = dnas.reshape(-1, *dna_shape).argmax(-1)
    else:
        flattened_dnas = dnas.reshape(-1, dna_shape[0])

    flattened_scores = scores.reshape(-1)
    flattened_measures = measures.reshape(-1, 2)

    metrics = jnp.concatenate([flattened_measures, flattened_scores[..., None]], axis=-1)

    # get coefficients
    alphas = np.geomspace(0.01, 1.0, 100)
    joint_lasso = MultiTaskLassoCV(alphas=alphas, max_iter=10000).fit(flattened_dnas, metrics)
    joint_coeffs = joint_lasso.coef_

    # plot
    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore

    fig, ax_joint = plt.subplots(figsize=(30, 10))
    ax_joint.set_yticks(range(len(descriptor_names) + 1))  # type: ignore
    ax_joint.set_yticklabels(list(descriptor_names) + ["score"])  # type: ignore

    ax_joint.set_xticks(range(flattened_dnas.shape[-1]))  # type: ignore
    ax_joint.set_xticklabels(range(flattened_dnas.shape[-1]))  # type: ignore

    vmax = np.abs(joint_coeffs).max()
    im = ax_joint.imshow(joint_coeffs, cmap='coolwarm', vmin=-vmax, vmax=vmax)  # type: ignore
    plt.colorbar(im, ax=ax_joint, pad=0.01)

    save_file = osp.join(save_dir, f"lasso_coefficients.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


def _compute_cross_decomposition(dnas, scores, measures, descriptor_names):
    # set up the data
    flattened_dnas = dnas.reshape(-1, dnas.shape[-1])
    flattened_scores = scores.reshape(-1)
    flattened_measures = measures.reshape(-1, measures.shape[-1])

    metrics = jnp.concatenate([flattened_measures, flattened_scores[..., None]], axis=-1)

    # get coefficients
    return PLSCanonical(n_components=len(descriptor_names) + 1).fit(flattened_dnas, metrics)


def dna_to_metric_cross_decomposition(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    dnas = model_outputs[0][0]
    input_is_distribution = model.dna_generator.return_raw_probabilities

    if input_is_distribution:
        dnas = dnas.argmax(-1)

    scores, measures = model_outputs[2][0][:2]  #type: ignore

    descriptor_names = trainer.task.problem.descriptor_names  # type: ignore
    coeffs = _compute_cross_decomposition(dnas, scores, measures, descriptor_names).coef_.T

    # plot
    fig, ax_joint = plt.subplots(figsize=(40, 10))
    ax_joint.set_yticks(range(len(descriptor_names) + 1))  # type: ignore
    ax_joint.set_yticklabels(list(descriptor_names) + ["score"])  # type: ignore

    ax_joint.set_xticks(range(dnas.shape[-1]))  # type: ignore
    ax_joint.set_xticklabels(range(dnas.shape[-1]))  # type: ignore

    vmax = np.abs(coeffs).max()
    im = ax_joint.imshow(coeffs, cmap='coolwarm', vmin=-vmax, vmax=vmax)  # type: ignore
    plt.colorbar(im, ax=ax_joint, pad=0.01)

    save_file = osp.join(save_dir, "cross-decomposition_coefficients.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


def dna_path_length_intervention(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
    n_coefficients=1,
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    dnas = model_outputs[0][0]

    if input_is_distribution:
        dnas = dnas.reshape(-1, *dna_shape).argmax(-1)

    scores, measures = model_outputs[2][0][:2]  #type: ignore

    # set up the data
    descriptor_names = trainer.task.problem.descriptor_names
    cross_decomp = _compute_cross_decomposition(dnas, scores, measures, descriptor_names)

    # select for path length for now, make a parameter for this later
    dna_char_rank = np.abs(cross_decomp.coef_.T)[0].argsort()

    # modify dna and see results
    best_path_length = measures[-1][:, 0].argmax()
    init_dna = dnas[-1, best_path_length]
    dna_char_idx = dna_char_rank[-n_coefficients:]  # argsort returns ascending order

    mutants = []
    for gene_substring in product(range(dna_shape[-1]), repeat=n_coefficients):
        gene_substring = jnp.asarray(gene_substring)
        if jnp.any(gene_substring != init_dna[dna_char_idx]):
            modified_dna = init_dna.at[dna_char_idx].set(gene_substring)
            mutants.append(modified_dna)

    mutants = jnp.stack(mutants)
    if input_is_distribution:
        mutants = jax.nn.one_hot(mutants, dna_shape[-1], axis=-1)

    # mutated_lvsl = jax.vmap(model.nca, in_axes=(0, None))(
    #     jax.nn.one_hot(mutants, dna_shape[-1], axis=-1), jr.key(3)
    # )[0]
    # mutant_scores, mutant_measures = jax.vmap(trainer.task.problem)(mutated_lvsl)[:2]  # type: ignore
    def eval_level(model_output):
        return trainer.task.problem(model_output[0])

    mutant_scores, mutant_measures = batched_eval(
        model.nca, mutants, eval_level, 4096, jr.key(1234)
    )[:2]

    fig, ax = plt.subplots(figsize=(10, 10))

    bd_min = trainer.task.problem.descriptor_min_val  # type: ignore
    bd_max = trainer.task.problem.descriptor_max_val  # type: ignore
    bd_names = trainer.task.problem.descriptor_names  # type: ignore

    ax.set_xlim((bd_min[0], bd_max[0]))  # type: ignore
    ax.set_ylim((bd_min[1], bd_max[1]))  # type: ignore
    ax.set_xlabel(bd_names[0])  # type: ignore
    ax.set_ylabel(bd_names[1])  # type: ignore

    ax.scatter(  # type: ignore
        x=measures[-1, best_path_length, 0],
        y=measures[-1, best_path_length, 1],
        c=scores[-1, best_path_length],
        label='original',
        marker='o',
        # vmin=, vmax=vmax
    )

    im = ax.scatter(  # type: ignore
        x=mutant_measures[:, 0],
        y=mutant_measures[:, 1],
        c=mutant_scores,
        label='mutated',
        marker='*',
        # vmin=vmin, vmax=vmax
    )

    plt.colorbar(im, ax=ax, pad=0.01)
    ax.legend()  # type: ignore

    save_file = osp.join(save_dir, f"dna_interventions-{n_coefficients}_coefficients.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)


def map_full_dna_space(
    model_outputs: PyTree,
    model: PyTree,
    trainer: Trainer,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    dna_shape = model.dna_generator.dna_shape
    input_is_distribution = model.dna_generator.return_raw_probabilities

    # modify dna and see results
    mutants = []
    for dna_string in product(range(dna_shape[-1]), repeat=dna_shape[0]):
        dna_string = jnp.asarray(dna_string)
        mutants.append(dna_string)

    mutants = jnp.stack(mutants)
    if input_is_distribution:
        mutants = jax.nn.one_hot(mutants, dna_shape[-1], axis=-1)

    # mutated_lvsl = jax.vmap(model.nca, in_axes=(0, None))(
    #     jax.nn.one_hot(mutants, dna_shape[-1], axis=-1), jr.key(3)
    # )[0]
    # mutant_scores, mutant_measures = jax.vmap(trainer.task.problem)(mutated_lvsl)[:2]  # type: ignore
    def eval_level(model_output):
        return trainer.task.problem(model_output[0])

    mutant_scores, mutant_measures = batched_eval(
        model.nca, mutants, eval_level, 4096, jr.key(1234)
    )[:2]

    fig, ax = plt.subplots(figsize=(10, 10))

    bd_min = trainer.task.problem.descriptor_min_val  # type: ignore
    bd_max = trainer.task.problem.descriptor_max_val  # type: ignore
    bd_names = trainer.task.problem.descriptor_names  # type: ignore

    ax.set_xlim((bd_min[0], bd_max[0]))  # type: ignore
    ax.set_ylim((bd_min[1], bd_max[1]))  # type: ignore
    ax.set_xlabel(bd_names[0])  # type: ignore
    ax.set_ylabel(bd_names[1])  # type: ignore

    im = ax.scatter(  # type: ignore
        x=mutant_measures[:, 0],
        y=mutant_measures[:, 1],
        c=mutant_scores,
        label='mutated',
        marker='*',
        # vmin=vmin, vmax=vmax
    )

    plt.colorbar(im, ax=ax, pad=0.01)
    ax.legend()  # type: ignore

    save_file = osp.join(save_dir, "dna_space.png")
    fig.savefig(save_file, bbox_inches='tight', dpi=300)  # type: ignore
    plt.close(fig)
