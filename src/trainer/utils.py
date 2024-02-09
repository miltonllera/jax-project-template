from dataclasses import dataclass, field
from heapq import heapify, heappush, heappop, heappushpop
from typing import Any, Union, Tuple

import jax


def aot_compilation(func, inputs, jit_kwargs=None):
    if jit_kwargs is None:
        jit_kwargs = {}

    return jax.jit(func, **jit_kwargs).lower(inputs, 0).compile()


@dataclass(order=True)
class PriorityItem:
    priority: Union[float, int]
    item: Any = field(compare=False)


class PriorityQueue:
    def __init__(self, max_cap: int, items):
        self.max_cap = max_cap
        self.items = [PriorityItem(*i) for i in items]
        heapify(self.items)
        while len(self.items) > max_cap:
            heappop(self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def lowest_priority(self):
        return self.items[0].priority

    def push_and_pop(self, item: Tuple[Union[float, int], Any]) -> Union[None, Tuple]:
        if len(self.items) == self.max_cap:
            value = heappushpop(self.items, PriorityItem(*item))
            value = value.priority, value.item
        else:
            heappush(self.items, PriorityItem(*item))
            value = None

        return value
