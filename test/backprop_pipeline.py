import pyrootutils
import logging

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,  # add to system path
    dotenv=True,      # load environment variables .env file
    cwd=True,         # change cwd to root
)

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.info('message')

import jax
import jax.random as jr
import equinox.nn as nn
import optax
from src.trainer.backprop import BackpropTrainer
from src.task.supervised import SupervisedTask
from src.dataset.emoji import SingleEmojiDataset
from src.model.dev import NCA
from src.nn.ca import (
    ConstantContextEncoder,
    IdentityControlFn,
    IdentityAndSobelFilter,
    SliceOutput,
    MaxPoolAlive
)


model = NCA(
    state_size=16,
    grid_size=(20, 20),
    dev_steps=(28, 48),
    update_prob=0.5,
    context_encoder=ConstantContextEncoder(16, (20, 20)),
    control_fn=IdentityControlFn(),
    alive_fn=MaxPoolAlive(alive_bit=3, alive_threshold=0.1),
    message_fn=IdentityAndSobelFilter(),
    update_fn=nn.Sequential(
        layers=[
            nn.Conv2d(in_channels=16 * 3, out_channels=32, kernel_size=1, key=jr.PRNGKey(1)),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, key=jr.PRNGKey(2)),
        ],
    ),
    output_decoder=SliceOutput(dim=0, start_idx=0, end_idx=3),
)

dataset = SingleEmojiDataset("salamander", 16, pad=2, batch_size=2)

task = SupervisedTask(dataset, lambda x, y: jax.numpy.power(x - y, 2).sum())

optim = optax.adam(0.01)

trainer = BackpropTrainer(task, optim, 1000, 5, 100)

trainer.run(model, jr.PRNGKey(0))
