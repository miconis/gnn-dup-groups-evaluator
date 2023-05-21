"""Module containing all the config."""
from typing import Literal
import torch

DEVICE: Literal['cpu', 'device'] = "cuda" if torch.cuda.is_available() else "cpu"
# parameters
EPOCHS: int = 400
BATCH_SIZE: int = 64
START_EPOCH: int = 1
BATCH_ACCUM: int = 1
LEARNING_RATE: int = 1e-4
EARLY_STOPPING: int = 20
