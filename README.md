
# Training 
In the training there are two types of models saved:
- `model_train_fold_{fold_id}.pt` : Its the best model evaluated by the validation set
- `best_model_fold_{fold_id}_epochs_{max_epochs}.pt` : Its the model trained with the optimal number of epochs and using the validation and training set to train the model.

# Transfer Learning with Pretrained Neural Network
To leverage the power of pretrained neural networks, consider incorporating transfer learning into your training process. Here's how you can integrate a pretrained neural network into your workflow:

1. **Choose a Pretrained Neural Network:**
   - Select a pretrained neural network that is well-suited for signal processing tasks. Popular choices include networks pretrained on image or audio data, which can capture useful hierarchical features.

2. **Load Pretrained Model:**
   - Load the pretrained neural network weights and architecture.

3. **Modify the Output Layer:**
   - Replace the original output layer with a new layer suitable for your specific signal processing task. This ensures the model adapts to your target classes.

4. **Freeze Pretrained Layers:**
   - Optionally, freeze the weights of the pretrained layers to prevent them from being updated during training. This is beneficial when the lower-level features are relevant to your task.

5. **Train the Model:**
   - Train the model on your signal dataset, possibly using the saved model formats mentioned earlier (`model_train_fold_{fold_id}.pt` and `best_model_fold_{fold_id}_epochs_{max_epochs}.pt`).

6. **Save Transfer Learning Models:**
   - Save models after transfer learning for later evaluation and use.
   
# Wandb config
In order to use the wandb plataform you must define two enviorement variables, `WANDB_KEY` and `WANDB_PROJECT`. These variables can be define in a `.env` file inside the same directory where the `main.py` will be executed. The `.env` template can be seen as follow.
```bash
WANDB_KEY="{user_wandb_key}"
WANDB_PROJECT="{wandb_project_name}"
```
