from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.animation import FuncAnimation


def generate_gif_from_array(
    frames: np.ndarray,
    fig: plt.FigureBase | None = None,  # type: ignore
    ax: plt.Axes | None = None,  # type: ignore
    cb_ax: plt.Axes | None = None,  # type: ignore
    vrange: tuple[float, float] = (0, 1),
    colorbar: bool = False,
    cmap: Colormap | None = None,
    colorbar_labels: Sequence[Any] | None = None,
):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax: plt.Axes = plt.gca()  # type: ignore

    # strip(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    im = ax.imshow(frames[0], cmap=cmap, aspect="auto", vmin=vrange[0], vmax=vrange[1])

    if colorbar:
        cb = plt.colorbar(im, ax=ax, cax=cb_ax, pad=0.01)
        if colorbar_labels is not None:
            cb.set_ticks(colorbar_labels)
            cb.set_ticklabels(colorbar_labels)

    def animate(i):
        # figures created using plt.figure will stay open and consume memory
        if i == len(frames):
            plt.close(fig)  # type: ignore
            return im,
        else:
            ax.set_xlabel(f"step: {i + 1}")
            im.set_array(frames[i])
            return im,

    return FuncAnimation(
        fig, animate, interval=200, blit=True, repeat=True, frames=len(frames) + 1  # type: ignore
    )
