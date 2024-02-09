import os
import subprocess

import torch
import wandb
from dotenv import load_dotenv

from vibnet.config import Config


class Experiment:
    def __init__(self, exp_name, output_dir: str = "output"):
        self.name = exp_name
        self.exp_dirpath = os.path.join(output_dir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, "models")
        self.results_dirpath = os.path.join(self.exp_dirpath, "results")
        self.cfg_path = os.path.join(self.exp_dirpath, "config.yaml")
        self.code_state_path = os.path.join(self.exp_dirpath, "code_state.txt")

        self.cfg: Config = None
        self.setup_exp_dir()

        # Load the enviroment variable from `.env`
        load_dotenv()
        # TODO: add loggin instead of printing in stdout
        # if args is not None:
        #     self.log_args(args)

    def set_cfg(self, cfg, override=False):
        assert "model_checkpoint_gap" in cfg
        self.cfg = cfg
        # Save the configuration used
        if not os.path.exists(self.cfg_path) or override:
            with open(self.cfg_path, "w") as cfg_file:
                cfg_file.write(str(cfg))

    def setup_exp_dir(self):
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            os.makedirs(self.results_dirpath)
            self.save_code_state()

    def save_code_state(self):
        state = "Git hash: {}".format(
            subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=False).stdout.decode("utf-8")
        )
        state += "\n*************\nGit diff:\n*************\n"
        state += subprocess.run(["git", "diff"], stdout=subprocess.PIPE, check=False).stdout.decode("utf-8")
        with open(self.code_state_path, "w") as code_state_file:
            code_state_file.write(state)

    def save_state(self, fold, epoch, model, optimizer, schedulers, file_name=None):
        if file_name is None:
            file_name = "model_fold_{:02d}_epochs_{:04d}.pt".format(fold, epoch)
        train_state_path = os.path.join(self.models_dirpath, file_name)
        # TODO:  add some hyperparameters into the state
        torch.save(
            {
                "fold": fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                **{f"scheduler{i}": scheduler.state_dict() for i, scheduler in enumerate(schedulers)},
            },
            train_state_path,
        )

    def get_results_dirpath(self):
        return self.results_dirpath

    def get_model_path(self, file_name: str):
        return os.path.join(self.models_dirpath, file_name)

    def get_model_state(self, model_path):
        return torch.load(model_path)["model"]

    def configure_wandb(self, run_name: str) -> None:
        if self.cfg is None:
            raise ValueError("In order to configure wandb run, the configuration must be set")
        # Retrieve global variables
        wandb.login(key=os.environ["WANDB_KEY"])
        wandb.init(
            # Set the project where this run will be logged
            project=os.environ["WANDB_PROJECT"],
            # Track essentials hyperparameters and run metadata
            config={
                "batch_size": self.cfg["batch_size"],
                "learning_rate": self.cfg["optimizer"]["parameters"]["lr"],
                "weight_decay": self.cfg["optimizer"]["parameters"]["weight_decay"],
                "epochs": self.cfg["epochs"],
                "arquitecture": self.cfg["model"]["name"],
            },
            # Set the name of the experiment
            name=run_name,
        )
        # Add configuration file into the wandb log
        wandb.save(self.cfg_path, policy="now")
        wandb.save(self.code_state_path, policy="now")
