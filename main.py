from argparse import Namespace, ArgumentParser
from datetime import datetime

from lib.config import Config
from lib.runner import ExpRunner
from lib.sampling import DataSampling
from lib.experiment import Experiment


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--epochs", help="Number of epochs", type=int)
    parser.add_argument("--cfg", help="Config file", required=True)
    parser.add_argument("--run", help="Name of the experiment", required=True)
    parser.add_argument("--lr", help="Learning rate", type=float)
    parser.add_argument("--pretrained", help="Pretrained", action="store_true")
    args = parser.parse_args()
    return args


def header_log(test_fold: int):
    print("".center(30, "="))
    print(f"Fold {test_fold}".center(30, "="))
    print("".center(30, "="))


def main():
    start_time = datetime.now()
    args = parse_args()
    exp = Experiment(args.run)

    cfg_path = args.cfg
    cfg = Config(cfg_path, args=args)

    # Configure the wandb and save the configuration used in the experiment
    exp.set_cfg(cfg, override=False)
    exp.configure_wandb(args.run)
    # Load dataset and datasampling
    dataset = cfg.get_dataset()
    data_sampling = DataSampling(dataset, cfg)

    runner = ExpRunner(data_sampling, cfg, exp)
    for test_fold in range(data_sampling.get_num_folds()):
        header_log(test_fold)
        # Allocate the test, val and train set based on the folds
        data_sampling.split(test_fold=test_fold, with_val_set=True)
        # Train the model with validation in order to find the best epoch
        best_epoch = runner.train(on_validation=True)
        # After finding the best epoch train with both training set and validation set until
        # the best epoch found
        runner.train(max_epochs=best_epoch)
        # Test with the model trained with both training and validation sets
        runner.eval(epoch=best_epoch)

    runner.finish()

    total_time = datetime.now() - start_time
    print("".center(30, "-"))
    print("\nExperiment finished with {} duration".format(total_time))


if __name__ == "__main__":
    main()
