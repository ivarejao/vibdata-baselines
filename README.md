# Vibnet
## Training
In the training there are two types of models saved:
- `model_train_fold_{fold_id}.pt` : Its the best model evaluated by the validation set
- `best_model_fold_{fold_id}_epochs_{max_epochs}.pt` : Its the model trained with the optimal number of epochs and using the validation and training set together to train the model.

## Wandb config
In order to use the wandb plataform you must define two enviorement variables, `WANDB_KEY` and `WANDB_PROJECT`. These variables can be define in a `.env` file inside the same directory where the `main.py` will be executed. The `.env` template can be seen as follow.
```bash
WANDB_KEY="{user_wandb_key}"
WANDB_PROJECT="{wandb_project_name}"
```



## Testing a Model

To test a model without training it, follow these steps:

1. Pass the `--test` flag to set the **testing mode**.
2. Use the `--model-dir` argument to specify the path where the models files are stored.
3. Note that you must include all other arguments used during model training, such as `batch-size`, `epochs` (if different from the config file), and other required arguments. These include the config file and the name of that run.

**Example Usage:**

```bash
python3 main.py --test --model-dir {path_to_model_dir} --dataset {dataset} --batch-size {bs_size} --cfg {config_path} --run evaluating/dataset
```
