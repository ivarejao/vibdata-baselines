# Training 
In the training there are two types of models saved:
- `model_train_fold_{fold_id}.pt` : Its the best model evaluated by the validation set
- `best_model_fold_{fold_id}_epochs_{max_epochs}.pt` : Its the model trained with the optimal number of epochs and using the validation and training set to train the model.

# Wandb config
In order to use the wandb plataform you must define two enviorement variables, `WANDB_KEY` and `WANDB_PROJECT`. These variables can be define in a `.env` file inside the same directory where the `main.py` will be executed. The `.env` template can be seen as follow.
```bash
WANDB_KEY="{user_wandb_key}"
WANDB_PROJECT="{wandb_project_name}"
```