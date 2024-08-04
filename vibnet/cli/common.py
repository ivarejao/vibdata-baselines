import os
import random
from enum import Enum
from typing import Any, Dict, Type

import numpy as np
import torch
import wandb
from dotenv import load_dotenv

from vibnet.data import group_dataset
from vibnet.data.rounds_repo import is_multi_round_dataset
from vibnet.data.group_dataset import GroupDataset

__all__ = ["set_deterministic", "wandb_login", "group_class", "is_logged", "Split"]

_is_logged = False


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


def is_logged():
    return _is_logged


def group_class(dataset_cfg: Dict[str, Any]) -> Type[GroupDataset]:
    group_obj_name = "Group{0}{1}".format(
        "MultiRound" if is_multi_round_dataset(dataset_cfg) else "", dataset_cfg["name"]
    )
    class_ = getattr(group_dataset, group_obj_name)
    return class_
