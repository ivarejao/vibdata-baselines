import os
import subprocess
from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv

# List of SSH connections with their name and the batchsize that each gpu can handle
HOSTS = {
    "pegasus": {"name": "pegasus", "batch_size": 60},
    "andromeda": {"name": "andromeda", "batch_size": 60},
    "nyx": {"name": "nyx", "batch_size": 28},
    "temis": {"name": "temis", "batch_size": 28},
}

DATASETS = [
    "PU",
    "MAFAULDA",
    "CWRU",
    "EAS",
    "MFPT",
    "UOC",
    "IMS",
    "XJTU",
]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--host", help="Host name", required=True, choices=list(HOSTS.keys())
    )
    parser.add_argument(
        "--dataset", help="Dataset to be used", required=True, choices=DATASETS
    )
    parser.add_argument(
        "--branch", help="Branch where the experiment will run", default="dev"
    )
    args = parser.parse_args()
    return args


def run_experiment(host_name, dataset, branch):
    # Define the environment variables
    env_variables = {
        "WANDB_PROJECT": os.environ["WANDB_PROJECT"],
        "WANDB_KEY": os.environ["WANDB_KEY"],
    }

    # Define the command to execute
    command_template = (
        "python3 main.py --cfg cfgs/xresnet18.yaml --run fullexp/1/{dataset}-{host_name}_run"
        + " --dataset {dataset} --batch-size {batch_size} --unbiased"
    )
    # # Iterate through SSH connections and dataset names
    # for idx, metadata in details.iterrows():
    bs = HOSTS[host_name]["batch_size"]
    try:
        ssh_cmd = (
            f"ssh {os.environ['SSH_USER']}@{host_name}"
            + " -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new"
        )

        checkout_cmd = f"cd vibnet && git fetch origin {branch} && git checkout {branch} && git pull origin {branch}"
        clone_cmd = (
            f"git clone -b {branch} https://{os.environ['GIT_USER']}:{os.environ['GIT_PASSWORD']}"
            + "@gitlab.com/ninfa-ufes/deep-rpdbcs/vibnet.git && cd vibnet"
        )
        # Setup de repository, if the repo already exists, only checkout into the target branch, otherwise,
        # clone the repo
        setup_repo = f"if [ -d vibnet/ ]; then {checkout_cmd}; else {clone_cmd}; fi"

        # Create .env file with environment variables
        env_content = "\n".join(
            [f"{key}={value}" for key, value in env_variables.items()]
        )
        env_cmd = f'echo "{env_content}" > .env'

        # Create python env and activate it
        pythonenv_create_cmd = "python3 -m venv vibenv"
        activate_cmd = "source vibenv/bin/activate"
        # Install dependencies
        dependecies_cmd = "pip install -r no_auth_requirements.txt"
        # Install vibdata package
        vibdata_install_cmd = (
            f"pip install git+https://{os.environ['GIT_USER']}:{os.environ['GIT_PASSWORD']}"
            + "@gitlab.com/ninfa-ufes/deep-rpdbcs/signal-datasets.git@e889f36eab3c016a9ab3c6dfb40ee85baa25e208"
        )
        # Execute the command with dataset and connection information
        run_cmd = command_template.format(
            dataset=dataset, host_name=host_name, batch_size=bs
        )

        cmds = [
            ssh_cmd,
            setup_repo,
            env_cmd,
            pythonenv_create_cmd,
            activate_cmd,
            dependecies_cmd,
            vibdata_install_cmd,
            run_cmd,
        ]
        full_command = (ssh_cmd + " " + " && ".join(cmds[1:])).split(" ")
        subprocess.run(full_command)

        print(f"Command executed for {dataset} on {host_name}.")
        print("---")
        print("\n".join(cmds))
        print()

    except subprocess.CalledProcessError as e:
        print(f"Error on {host_name}: {e}")


def main():
    args = parse_args()
    # Load the environment variables
    load_dotenv(".env_exp")
    dataset = args.dataset
    host_name = args.host
    branch = args.branch
    run_experiment(host_name, dataset, branch)


if __name__ == "__main__":
    main()
