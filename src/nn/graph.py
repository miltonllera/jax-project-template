import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from typing import Callable, NamedTuple, Optional
from jaxtyping import Array, Float

from src.utils import Tensor


# Graph = Tuple[Tensor, Tensor, Tensor, Tensor]  # Node embedding, edge embeddings, adjacency matrix
class Graph(NamedTuple):
    """
    Representation of a graph.

    Optionally, these can have edge embeddings. Nodes can be added up to some maximum capacity.
    The ones currently used are specified by the mask attribute. We assume that the adjacency
    matrix already encodes the absence of edges to and from unused nodes.
    """
    nodes: Float[Array, "N F"]
    adj: Float[Array, "N N 1"]
    mask: Float[Array, "N 1"]
    edges: Float[Array, "N N E"]


class GraphAttentionLayer(eqx.Module):
    """
    Implements the Graph Attention layer (aka GAT) as described in "Graph Attention Networks",
    Petar Veličković et al., (2017):

        https://arxiv.org/abs/1710.10903

    This layer uses an MLP to compute the attention logits instead of the now more standard
    dot-product attention. It also has no normalisation operations applied to the result. The
    addition of position embeddings is left to a different class.

    TODO: It should be possible to merge this class with the one below using dot-product attention.
    """
    attn_fn: nn.MLP
    preW: nn.Linear
    postW: nn.Linear
    n_heads: int
    use_edges_features: bool
    use_adj_matrix_as_mask: bool
    sum_heads: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        n_heads: int = 1,
        attn_depth: int = 0,
        attn_width: Optional[int]=None,
        use_edge_features: bool = False,
        use_adj_matrix_as_mask: bool = True,
        sum_heads: bool = False,
        *,
        key: jax.Array,
    ):
        if attn_width is None:
            attn_width = 3 * in_dim

        if attn_depth > 0 and attn_width == 0:
            raise RuntimeError(
                "In GAT layer: attention depth is greater than 0, but MLP width is set to 0."
            )

        if use_edge_features and edge_dim == 0:
            raise RuntimeError("In GAT layer: use_edges set to True, but edge_features is zero.")

        preW_key, attn_key, postW_key = jr.split(key, 3)

        self.preW = nn.Linear(in_dim, out_dim, use_bias=False, key=preW_key)
        self.postW = nn.Linear(in_dim, out_dim * n_heads, use_bias=False, key=postW_key)

        self.attn_fn = nn.MLP(
            2 * out_dim + (edge_dim if use_edge_features else 0),
            n_heads,
            attn_width,
            attn_depth,
            key=attn_key,
            activation=jnn.leaky_relu,
            final_activation=jnn.leaky_relu,
        )

        self.n_heads = n_heads
        self.use_edges_features = use_edge_features
        self.use_adj_matrix_as_mask = use_adj_matrix_as_mask
        self.sum_heads = sum_heads

    def __call__(self, graph: Graph, key: jax.Array) -> Graph:
        h, a, m, e = graph
        N = h.shape[0]

        # notice that we don't need to multiply by the mask when projecting since the bias is false
        # this assumes that the embedding for unused nodes is the null vector.
        h_proj = jax.vmap(self.preW)(h)

        W = jnp.concatenate([
            jnp.repeat(h_proj[:, None], N, axis=1),
            jnp.repeat(h_proj[None], N, axis=0)
        ], axis=-1) # N x N x 2F

        if self.use_edges_features:
            assert e is not None
            W = jnp.concatenate([W, e], axis=-1)  # N x N x (2F + E)

        # compute the attention for each entry in the N x N grid, thus double vmap
        attn_w = jax.vmap(jax.vmap(self.attn_fn))(W)  # N x N x n_head

        # NOTE: the adjecency matrix should account for unused nodes, but if it is not used we
        # must account for this issue manually.
        where = a if self.use_adj_matrix_as_mask else jnp.dot(m, m.T)
        initial = jnp.min(attn_w * where)
        attn_w = jnn.softmax(attn_w, where=where, axis=1, initial=initial)

        out_h = jax.vmap(self.postW)(h).reshape((N, -1, self.n_heads))  # N, F_out, n_heads
        out_h = jnp.einsum("nN...,Nf...->nf...", attn_w, out_h)  # N, F_out, n_heads
        out_h = out_h.sum(-1) if self.sum_heads else out_h.reshape(N, -1)

        return graph._replace(nodes=out_h)


class GraphNonLinearity(eqx.Module):
    non_linearity: Callable[[Tensor], Tensor]
    apply_to_edges: bool = False

    def __call__(self, inputs: Graph, key: jax.Array) -> Graph:
        h = inputs.nodes
        h = self.non_linearity(h)

        if self.apply_to_edges:
            e = inputs.edges
            assert e is not None
            e = self.non_linearity(e)
            return inputs._replace(nodes=h, edges=e)

        return inputs._replace(nodes=h)


linalg_norm = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class GraphTransformerLayer(eqx.Module):
    """
    Implements the Graph Transformer layer as described in "A Generalization of Transformer
    Networks to Graphs", Dwivedi & Bresson (2021):

        https://arxiv.org/pdf/2012.09699v2.pdf

    This is very similar to the GraphAttentionLayer except:
        1. It uses dot-product attention instead of an MLP.
        2. Applies residual connections and normalisation to the output. Note that this last
           feature is disabled by default since we are using this in an evolutionary setting where
           gradient flow is not a concern.
        3. There is no out_dim, it is the same as in_dim. Expansion/contraction of embeddings must
           be done in a separate layer.

    Also, we leave the addition of the position embeddings to a different class. This allows us to
    use different types of position embedding apart from the original Laplacian Eigen-vectors.

    TODO: It should be possible to unify these two algorithms into one class.

    """
    # params :
    Q: nn.Linear   # Query function
    K: nn.Linear   # Key
    V: nn.Linear   # Value
    node_out_proj: nn.Linear   # Output V*heads->do
    node_norm: Callable[[Tensor], Tensor]
    node_residual_mlp: Optional[nn.Sequential]
    edge_proj: Optional[nn.Linear]
    edge_out_proj: Optional[nn.Linear]
    edge_norm: Optional[Callable[[Tensor], Tensor]]
    edge_residual_mlp: Optional[nn.Sequential]
    # statics :
    n_heads: int
    use_edge_features: bool
    qk_dim: int
    value_features: int

    def __init__(
        self,
		node_dim: int,
		qk_dim: Optional[int] = None,
		value_dim: Optional[int] = None,
		n_heads: int = 1,
		use_edge_features: bool=False,
		edge_dim: int=1,
        norm_layer: Optional[Callable[[int], Callable[[Tensor], Tensor]]] = None,
		use_bias: bool=False,
        use_residual_mlps: bool = False,  # not necessary when doing evo training
        res_mlp_width_factor: Optional[int] = None,
        *,
		key: jax.Array
    ):
        if qk_dim is None:
            qk_dim = node_dim
        if value_dim is None:
            value_dim = node_dim

        key_Q, key_K, key_V, key_O_e, key_E_w, key_E_o, key = jr.split(key, 7)

        self.n_heads = n_heads
        self.use_edge_features = use_edge_features
        self.qk_dim = qk_dim
        self.value_features = value_dim

        if res_mlp_width_factor is None:
            res_mlp_width_factor = 2

        self.Q = nn.Linear(node_dim, qk_dim * n_heads, key=key_Q, use_bias=use_bias)
        self.K = nn.Linear(node_dim, qk_dim * n_heads, key=key_K, use_bias=use_bias)
        self.V = nn.Linear(node_dim, value_dim * n_heads, key=key_V, use_bias=use_bias)
        self.node_out_proj = nn.Linear(value_dim * n_heads, node_dim, key=key_O_e)

        if norm_layer is None:
            self.node_norm = linalg_norm
        else:
            self.node_norm = norm_layer(node_dim)

        # NOTE: Erwan removed the residual MLPs. These are not necessary in an evolutionary setting
        # as they are usally added to help with gradients flowing backwards. To keep it consistent
        # I have added an option to toggle them off.
        if use_residual_mlps:
            key1, key2, key = jr.split(key, 3)
            self.node_residual_mlp = nn.Sequential(
                nn.Linear(node_dim, res_mlp_width_factor * node_dim, use_bias=False, key=key1),
                nn.Lambda(jnn.relu), # type: ignore
                nn.Linear(res_mlp_width_factor * node_dim, node_dim, use_bias=False, key=key2)
            )
        else:
            self.node_residual_mlp = None

        if use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, n_heads, key=key_E_w, use_bias=use_bias)
            self.edge_out_proj = nn.Linear(n_heads, edge_dim, key=key_E_o, use_bias=use_bias)

            if norm_layer is None:
                self.edge_norm = linalg_norm
            else:
                self.edge_norm = norm_layer(edge_dim)

            if use_residual_mlps:
                key1, key2, key = jr.split(key, 3)
                self.edge_residual_mlp = nn.Sequential(
                    nn.Linear(edge_dim, res_mlp_width_factor * edge_dim, use_bias=False, key=key1),
                    nn.Lambda(jnn.relu),  # type: ignore
                    nn.Linear(res_mlp_width_factor * edge_dim, edge_dim, use_bias=False, key=key2)
                )
            else:
                self.edge_residual_mlp = None
        else:
            self.edge_proj, self.edge_out_proj, self.edge_residual_mlp = None, None, None

    def __call__(self, graph: Graph, *, key: jax.Array = None) -> Graph:
        n, adj, _, e = graph
        N = n.shape[0]

        q = jax.vmap(self.Q)(n).reshape(N, self.qk_dim, -1)  # N x qk_dim x H
        k = jax.vmap(self.K)(n).reshape(N, self.qk_dim, -1)  # N x qk_dim x H
        v = jax.vmap(self.V)(n).reshape(N, self.qk_dim, -1)  # N x value_dim x H

        # Compute attention scores (before softmax) (N x N x H)
        scores = jnp.einsum("ndh,Ndh->nNh", q, k) / jnp.sqrt(self.qk_dim)

        if self.use_edge_features:
            assert (
                e is not None and self.edge_proj is not None and
                self.edge_out_proj is not None and self.edge_norm is not None
            )

            we = jax.vmap(jax.vmap(self.edge_proj))(e) * adj
            scores = scores * we

            e = self.edge_norm(e + jax.vmap(jax.vmap(self.edge_out_proj))(scores))
            if self.edge_residual_mlp is not None:
                e = self.edge_norm(we + jax.vmap(jax.vmap(self.edge_residual_mlp))(e))

            graph = graph._replace(edges=e)

        # attn weights
        attn_w = jnn.softmax(scores, axis=1, where=adj, initial=jnp.min(scores * adj))

        h = jnp.einsum("nNh,Ndh->ndh", attn_w, v).reshape(N, -1)

        # TODO: Make this normalization function a parameter of the model
        n = self.node_norm(n + jax.vmap(self.node_out_proj)(h))
        if self.node_residual_mlp is not None:
            n = self.node_norm(n + jax.vmap(self.node_residual_mlp)(n))

        return graph._replace(nodes=n)
