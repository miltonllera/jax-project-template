import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


@jax.jit
def to_adjecency(nodes: Float[Array, "N"], connections: Float[Array, "M 2"]) -> Bool[Array, "N N"]:
    adjacency = jnp.zeros((nodes, nodes), dtype=bool)
    return adjacency.at[connections.T].set(True)


@jax.jit
def check_path(
    nodes: Float[Array, "N"],
    connections: Float[Array, "M 2"],
    start: int,
    end: int
) -> Bool[Array, "1"]:
    # TODO: Find a way to do this without having to instantiate the full adjacency matrix
    adjacency = to_adjecency(nodes, connections)

    def loop_cond(carry):
        visited_changed, visited = carry
        return  visited[end] | ~visited_changed

    def loop_body(carry):
        _, visited = carry
        new_visited = adjacency.T @ visited  # dot product with boolean matrices performs an OR
        visited_changed = jnp.any(visited != new_visited)
        return visited_changed, visited

    visited = jnp.zeros(nodes.shape[0], dtype=bool).at[start].set(True)
    visited = jax.lax.while_loop(loop_cond, loop_body, (jnp.array([True]), visited))[1]
    return visited[end]
