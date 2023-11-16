# Experiment Runner Tool

The Experiment Runner Tool is a Python script designed to facilitate running experiments from a repository on a specific host. This README provides step-by-step instructions on setting up and using the tool.

## Setup

### 1. Environment Variables

Create a `.env_exp` file inside the root directory of the repository and specify the following environment variables:

- `WANDB_PROJECT`: The project of your Weights & Biases (wandb) account.
- `WANDB_KEY`: The access key to connect the experiment with your wandb project.
- `GIT_USER`: The username of your GitLab account.
- `GIT_PASSWORD`: The password or an access token of your GitLab account.

The first two variables are used to connect to the wandb platform, and the other two are to clone the experiment repository.

### 2. SSH Key Setup

Ensure that the machine running the tool is in the authorized keys on the host. Copy your public SSH key into the host using the following command:

```bash
ssh-copy-id user@host
```

Repeat this step for every host you plan to run experiments on.

### 3. Internet Connection Check

Ensure that the host has an internet connection by logging into `pppoedi`.

### 4. Virtual enviroment creation

Create a virtual enviroment and install requirements in `tools/requirements.txt`

## Running Experiments

After setting up the tool, follow these steps to run an experiment:

1. Create a tmux session.
2. Activate the virtual environment.
3. Run the tool from the repository root:

```bash
python3 tools/exprunner --host {host_name} --dataset {dataset_name}
```

### Notes
##### Branch

If you want to use a different branch from the default (`dev`), you can specify the argument `--branch`.

##### Batch Size

The `batch_size` specified in the program was computed based on the GPU capacity of each host and by the experiment using interpolation. Your experiment may probably use a different size.
