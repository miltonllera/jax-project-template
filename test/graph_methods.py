import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)

# import jax
import jax.numpy as jnp
import jax.random as jr
from src.nn.graph import Graph, GraphAttentionLayer, GraphTransformerLayer
from src.utils.tree import tree_shape

N, node_dim, edge_dim, out_dim = 10, 4, 3, 8

if __name__ == "__main__":
    mask = jnp.ones((N, 1))  # vary this to create graphs with unused nodes.
    nodes = jr.uniform(jr.key(1), shape=(N, node_dim), minval=-1.0, maxval=1.0) * mask
    adj = (jr.bernoulli(jr.key(2), p=0.5, shape=(N, N)) * jnp.dot(mask, mask.T))[..., None]
    edges = jr.uniform(jr.key(3), shape=(N, N, edge_dim), minval=-1.0, maxval=1.0) * adj

    graph = Graph(nodes=nodes, adj=adj, mask=mask, edges=edges)

    # gat = GraphAttentionLayer(
    #     in_dim=node_dim,
    #     out_dim=out_dim,
    #     edge_dim=edge_dim,
    #     n_heads=2,
    #     attn_depth=0,
    #     attn_width=None,
    #     use_edge_features=True,
    #     use_adj_matrix_as_mask=True,
    #     sum_heads=True,
    #     key=jr.key(0)
    # )
    gtl = GraphTransformerLayer(
        node_dim=node_dim,
        edge_dim=edge_dim,
        n_heads=2,
        use_edge_features=True,
        key=jr.key(0)
    )

    print("original graph:\n", tree_shape(graph))
    # transformed_graph = gat(graph, key=jr.key(42))
    transformed_graph = gtl(graph, key=jr.key(42))
    print("transformed graph:\n", tree_shape(transformed_graph))
