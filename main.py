from argparse import Namespace, ArgumentParser
from datetime import datetime

from lib.config import Config
from lib.runner import ExpRunner
from lib.sampling import DataSampling
from lib.experiment import Experiment


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Config file", required=True)
    parser.add_argument("--run", help="Name of the experiment", required=True)
    parser.add_argument("--classifier", help="Classifier of the model", required=True)
    parser.add_argument("--dataset", help="The dataset name")
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
        data_sampling.split(test_fold=test_fold)
        runner.train()
        runner.eval()

    runner.finish()

    total_time = datetime.now() - start_time
    print("".center(30, "-"))
    print("\nExperiment finished with {} duration".format(total_time))


if __name__ == "__main__":
    main()
