# Training 
In the training there are two types of models saved:
- `model_train_fold_{fold_id}.pt` : Its the best model evaluated by the validation set
- `best_model_fold_{fold_id}_epochs_{max_epochs}.pt` : Its the model trained with the optimal number of epochs and using the validation and training set to train the model.