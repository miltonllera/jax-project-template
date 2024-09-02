import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def get_layout(layout):
    return getattr(nx.layout, layout)


def get_layer_positions(n_nodes):
    positions = np.array([i + 0.5 for i in range(n_nodes)])
    if len(positions) > 0:
        return nx.rescale_layout(positions, n_nodes)
    return positions


def plot_graph(graph, positions=None, labels=None, layout=None, options=None):
    if options is None:
        options = {}

    if layout is not None:
        positions = get_layout(layout)(graph, pos=positions)

    fig, ax = plt.subplots(figsize=(20, 10))

    nx.draw_networkx(graph, positions, labels=labels, **options, ax=ax)
    ax.margins(0.20)
    ax.axis("off")

    return fig
