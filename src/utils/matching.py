from chex import Numeric
import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.solvers import linear
from jaxtyping import Array, Float


def optimal_match(cost_matrix):
    """
    Solves the matching problem for a given cost matrix using the Sinkhorn algorithm.
    This will return a coupling matrix where each entry is the strength of the match between
    a pair of points. We can the iteratively select the best match for each row and column.
    """
    geom = geometry.Geometry(cost_matrix)
    cost = linear.solve(geom).matrix

    # greedily select rows and columns based on coupling strength
    min_val = cost.min()

    def step(cost, i):
        j = cost[i].argmax()
        cost = cost.at[:, j].set(min_val - 1)
        return cost, j

    rows = jnp.argsort(cost.max(axis=1), descending=True)
    cols = jax.lax.scan(step, cost, rows)[1]

    return jnp.stack([rows, cols], axis=-1)[rows]  # revert the argsort by indexing it with itself


def intersection1d(
    arr1: Float[Array, "N"],
    arr2: Float[Array, "M"],
    size: int | None = None,
    fill_value: Float[Array, "1"] | None = None,
    assume_unique: bool = False,
    assume_ordered: bool = False
) -> Float[Array, "I"]:
    if not assume_unique:
        raise NotImplementedError

    if not assume_ordered:
        arr1 = jnp.sort(arr1, axis=0)
        arr2 = jnp.sort(arr2, axis=0)

    if size is None:
        size = min(arr1.shape[0], arr2.shape[0])

    if fill_value is None:
        fill_value = jnp.maximum(arr1.max(), arr1.max())

    intersection = jnp.full((size,), fill_value)

    def update_intersection(inter, i_head, a1_head, a2_head):
        return inter.at[i_head].set(arr1[a1_head]), i_head + 1, a1_head + 1, a2_head + 1

    def move_heads_only(inter, i_head, a1_head, a2_head):
        a1_head, a2_head = jax.lax.cond(
            arr1[a1_head] < arr2[a2_head],
            lambda a1h, a2h: (a1h + 1, a2h),
            lambda a1h, a2h: (a1h, a2h + 1),
            a1_head, a2_head
        )
        return inter, i_head, a1_head, a2_head


    def update(carry):
        inter, i_head, a1_head, a2_head = carry

        inter, i_head, a1_head, a2_head = jax.lax.cond(
            arr1[a1_head] == arr2[a2_head],
            update_intersection,
            move_heads_only,
            inter, i_head, a1_head, a2_head
        )

        carry = inter, i_head, a1_head, a2_head
        return carry

    intersection = jax.lax.while_loop(
        lambda carry: (carry[2] < len(arr1)) & (carry[3] < len(arr2)),
        update,
        (intersection, 0, 0, 0),
    )[0]

    return intersection
