import os
from pathlib import Path
from argparse import Namespace, ArgumentParser
from datetime import datetime

import numpy as np

from lib.config import Config
from lib.runner import ExpRunner
from lib.sampling import DataSampling
from lib.experiment import Experiment


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Config file", required=True)
    parser.add_argument("--run", help="Name of the experiment", required=True)
    parser.add_argument("--epochs", help="Number of epochs", type=int)
    parser.add_argument("--batch-size", help="Size of minibatching", type=int)
    parser.add_argument("--lr", help="Learning rate", type=float)
    parser.add_argument("--dataset", help="The dataset name")
    test_group = parser.add_argument_group(
        "Testing model", "These arguments should be setted when the goal is to test a specific model"
    )
    test_group.add_argument("--test", help="If the script should only test the model", action="store_true")
    test_group.add_argument(
        "--model-dir",
        default=None,
        type=str,
        help="The path to the directory where the models files are stored, e.g (`best_model_{fold}*.pt`)",
    )
    args = parser.parse_args()
    return args


def header_log(test_fold: int):
    print("".center(30, "="))
    print(f"Fold {test_fold}".center(30, "="))
    print("".center(30, "="))


def main():
    start_time = datetime.now()
    args = parse_args()
    test = args.test
    exp = Experiment(args.run)

    cfg_path = args.cfg
    cfg = Config(cfg_path, args=args)

    # Configure the wandb and save the configuration used in the experiment
    exp.set_cfg(cfg, override=False)
    exp.configure_wandb(args.run)
    # Load dataset and datasampling
    dataset = cfg.get_dataset()
    data_sampling = DataSampling(dataset, cfg)

    # If only need to test, list the models directory obtaining the models name for each fold
    if test:
        model_dir = Path(args.model_dir)
        models_paths = set(
            [str(model_dir / f.name) for f in model_dir.iterdir() if f.is_file() and ("best_model" in f.name)],
        )

    runner = ExpRunner(data_sampling, cfg, exp)
    for test_fold in range(data_sampling.get_num_folds()):
        header_log(test_fold)
        # Allocate the test, val and train set based on the folds
        data_sampling.split(test_fold=test_fold, with_val_set=True)
        if test:
            model_path = [path for path in models_paths if "best_model_fold_{:02d}".format(test_fold) in path][0]
            runner.eval(model_fname=model_path, complete_path=True)
        else:
            # Train the model with validation in order to find the best epoch
            best_epoch, best_params = runner.grid_search_train()
            # After finding the best epoch train with both training set and validation set until
            # the best epoch found
            data_sampling.split(test_fold=test_fold, with_val_set=False)
            runner.train(max_epochs=best_epoch, **best_params)
            # Test with the model trained with both training and validation sets
            model_fname = "best_model_fold_{:02d}_epochs_{:03d}.pt".format(test_fold, best_epoch)
            runner.eval(model_fname=model_fname)

    runner.finish()

    total_time = datetime.now() - start_time
    print("".center(30, "-"))
    print("\nExperiment finished with {} duration".format(total_time))


if __name__ == "__main__":
    main()
