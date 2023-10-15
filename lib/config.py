import os

import yaml
import torch
import vibdata.raw as datasets
import vibdata.deep.signal.transforms as deep_transforms
from vibdata.deep.DeepDataset import DeepDataset, convertDataset

from lib.models.model import Model


class Config:
    def __init__(self, config_path, args=None):
        self.config = {}
        self.load(config_path)
        name = self.config["model"]["name"]
        parameters = self.config["model"]["parameters"]
        self.model_constructor = Model(name, **parameters)

        # Override the configuration with the cli args
        if args:
            self.args = args
            self.config["epochs"] = self.config["epochs"] if args.epochs is None else self.args.epochs
            self.config["optimizer"]["parameters"]["lr"] = (
                self.config["optimizer"]["parameters"]["lr"] if args.lr is None else self.args.lr
            )

    def load(self, path):
        with open(path, "r") as file:
            config_str = file.read()
        self.config = yaml.load(config_str, Loader=yaml.FullLoader)

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config

    def get_yaml(self):
        return self.config

    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config["optimizer"]["name"])(
            model_parameters, **self.config["optimizer"]["parameters"]
        )

    def get_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        lr_schedulers = []
        for scheduler in self.config["lr_scheduler"]:
            lr_schedulers.append(
                getattr(torch.optim.lr_scheduler, scheduler["name"])(optimizer, **scheduler["parameters"])
            )
        return lr_schedulers

    def get_device(self):
        """
        If cuda device is avalaible, get the last one, as it more likely to not been used by other users
        """
        if torch.cuda.is_available():
            dev_number = torch.cuda.device_count()
            device = torch.device(f"cuda:{dev_number-1}")
        else:
            device = torch.device("cpu")
        return device

    def get_model(self, **kwargs) -> torch.nn.Module:
        # Besides the default parameters given when Config is instantiated, these can be override, passing as kwargs
        # into this method
        return self.model_constructor.new(**kwargs)

    def get_dataset(self):
        dataset_name = self.config["dataset"]["name"]

        # Get raw root_dir
        raw_root_dir = self.config["dataset"]["raw"]["root"]
        raw_dataset_package = getattr(datasets, dataset_name)
        raw_dataset_module = getattr(raw_dataset_package, dataset_name)
        raw_dataset = getattr(raw_dataset_module, dataset_name + "_raw")(raw_root_dir)

        deep_root_dir = os.path.join(self.config["dataset"]["deep"]["root"], dataset_name)
        # Get the transforms to be applied
        transforms_config = self.config["dataset"]["deep"]["transforms"]
        transforms = [getattr(deep_transforms, t["name"])(**t["parameters"]) for t in transforms_config]
        # Convert the raw dataset to deepdataset
        convertDataset(dataset=raw_dataset, transforms=transforms, dir_path=deep_root_dir)
        dataset = DeepDataset(deep_root_dir)
        return dataset
