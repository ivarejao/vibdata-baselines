import os
import random
from enum import Enum
from typing import Type

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

from vibnet.data import group_dataset
from vibnet.data.group_dataset import GroupMirrorBiased

__all__ = ["set_deterministic", "wandb_login", "group_class", "is_logged", "GroupMirrorBiased", "Split"]

_is_logged = False

TOTAL_SPLITS = 30


class Split(str, Enum):
    biased_usual = ("biased_usual",)
    biased_mirrored = ("biased_mirrored",)
    unbiased = "unbiased"
    multi_round = "multi_round"


def set_deterministic(seed: int):
    # Fix seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # CUDA convolution determinism
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Set cubLAS enviroment variable to guarantee a deterministc behaviour in multiple streams
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def wandb_login():
    global _is_logged

    load_dotenv()
    key = os.environ.get("WANDB_KEY", "")
    if key == "":
        raise RuntimeError("wandb must be set to use logging")
    wandb.login(key=key)
    _is_logged = True


def group_class(dataset_name: str, split: Split) -> Type[group_dataset.GroupDataset]:
    group_obj_name = "Group{0}{1}".format("MultiRound" if split is Split.multi_round else "", dataset_name)
    class_ = getattr(group_dataset, group_obj_name)
    return class_


def is_logged():
    return _is_logged
