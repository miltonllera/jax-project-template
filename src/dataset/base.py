import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate


def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))


class NumpyLoader(DataLoader):
  def __init__(
      self,
      dataset,
      batch_size=1,
      shuffle=False,
      sampler=None,
      batch_sampler=None,
      num_workers=0,
      pin_memory=False,
      drop_last=False,
      timeout=0,
      worker_init_fn=None
  ):
    super().__init__(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn
    )
