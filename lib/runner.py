import os
import sys
import random
import itertools
from typing import Any, Dict, Tuple

import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

import wandb
from lib.config import Config
from lib.sampling import DataSampling
from lib.experiment import Experiment
from lib.utils.report import ReportDict
from lib.models.model import Model

from sklearn.ensemble import RandomForestClassifier


class ExpRunner:
    def __init__(
        self,
        data_sampling: DataSampling,
        config: Config,
        experiment: Experiment,
        deterministic: bool = True,
    ) -> None:
        self.dataset = data_sampling
        self.config = config
        self.model = None
        self.experiment = experiment
        self.data_report = ReportDict(["real_label", "predicted", "fold", "dataset"])

        self.classifier = None

        if deterministic:
            self.set_deterministic(self.config["seed"])

    def _model_train(self, on_validation: bool = False, max_epochs: int = None, **kwargs):
        device = self.config.get_device()
        model = self.config.get_model()
        model.to(device)
        model.apply(Model.reset_weights)
        optimizer = self.config.get_optimizer(model_parameters=model.parameters(), **kwargs)
        train_loader = self.dataset.get_trainloader()
        schedulers = self.config.get_lr_scheduler(optimizer=optimizer)
        test_fold = self.dataset.get_fold()

        if on_validation:
            # Reset the best validation loss
            best_validation_loss = sys.maxsize
            best_train_epoch = 0

        criterion = nn.CrossEntropyLoss()

        starting_epoch = 1
        max_epochs = self.config["epochs"] if max_epochs is None else max_epochs
        prefix_log = f"fold:{test_fold}/" if on_validation else f"fold:{test_fold}/final_"

        scaler = GradScaler()
        model.train()
        for epoch in range(starting_epoch, max_epochs + 1):
            # Train the net
            train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.float().to(device)

                labels = labels.to(device)
                labels -= self.dataset.get_labels().min()  # Normalize the labels

                optimizer.zero_grad()  # Zero the graidients
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)  # Perform forward pass
                    loss = criterion(outputs, labels)  # Compute loss

                scaler.scale(loss).backward()  # Do the backpropagation
                scaler.step(optimizer)  # Update the weights
                scaler.update()

                train_loss += float(loss)

            # Validate
            train_loss = train_loss / (i + 1)

            # Define logs messages
            wandb_metrics_log = {
                f"{prefix_log}train_loss": train_loss,
                f"{prefix_log}learning_rate": optimizer.param_groups[0]["lr"],
            }

            log_message = f"[{epoch : 3d}] train_loss: {train_loss:5.3f} "

            model_fname = "model_train_fold_{:02d}.pt".format(test_fold)
            self.experiment.save_state(test_fold, epoch, model, optimizer, schedulers, file_name=model_fname)

            if on_validation:
                val_loss = self.eval(model_fname, on_validation=True)

                wandb_metrics_log.update({f"{prefix_log}val_loss": val_loss})
                log_message += f"| val_loss: {val_loss:5.3f}"

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    best_train_epoch = epoch

            wandb.log(wandb_metrics_log)
            print(log_message)

            # Update the learning rate
            for s in schedulers:
                s.step()

        # If it was only training the net without searching for the best parameters, save the current model
        # at the end of the training as the `best_model_{fold}`
        if not on_validation:
            self.experiment.save_state(
                test_fold,
                max_epochs,
                model,
                optimizer,
                schedulers,
                file_name="best_model_fold_{:02d}_epochs_{:03d}.pt".format(test_fold, max_epochs),
            )

        if on_validation:
            ret = (best_train_epoch, best_validation_loss)
        else:
            ret = (max_epochs, None)
        return ret

    def _classifier_train(self, on_validation: bool = False, **kwargs):
        train_loader = self.dataset.get_trainloader()
        classifier = RandomForestClassifier(n_jobs=-1, **kwargs)

        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels -= self.dataset.get_labels().min()  # Normalize the labels
            classifier = classifier.fit(inputs, labels)

        if on_validation:
            self.classifier = classifier
            val_loss = self.eval(None, on_validation=True)
            return (0, val_loss)
        return (0, None)

    def train(self, on_validation: bool = False, max_epochs: int = None, **kwargs):
        training_name = "Training" if on_validation else "Final Training"
        print(training_name.center(30, "="))

        if "classifier" in self.config.config:
            return self._classifier_train(on_validation, **kwargs)
        else:
            return self._model_train(on_validation, max_epochs, **kwargs)

    def _eval_report(self, y_pred: np.ndarray, y_true: np.ndarray, test_fold):
        # Creates table from classification report and log it
        data = {"predicted": y_pred, "real_label": y_true}
        eval_size = len(data["predicted"])
        # Add the test data into the final report
        self.data_report.update(
            data,
            fold=[
                test_fold,
            ]
            * eval_size,
            dataset=[
                "test",
            ]
            * eval_size,
        )
        # Report in the stdout
        class_report = classification_report(
            data["real_label"],
            data["predicted"],
            target_names=self.dataset.get_labels_name(),
            zero_division=0,
        )
        print(class_report)

        bal_acc = balanced_accuracy_score(data["real_label"], data["predicted"]) * 100
        print("\nBalanced Accuracy: {:.3f}%".format(bal_acc))
        wandb.run.summary[f"{test_fold}_balanced_accuracy"] = bal_acc

        f1_macro = f1_score(data["real_label"], data["predicted"], average="macro") * 100
        print("F1 Macro: {:.3f}%".format(f1_macro))
        wandb.run.summary[f"{test_fold}_f1_macro"] = f1_macro

        acc = accuracy_score(data["real_label"], data["predicted"]) * 100
        print("Accuracy: {:.3f}%".format(acc))
        wandb.run.summary[f"{test_fold}_accuracy"] = acc

        y_true_ = y_true - min(y_true)
        y_pred_ = y_pred - min(y_pred)

        wandb.log(
            {
                f"{test_fold}_conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_,
                    preds=y_pred_,
                    class_names=self.dataset.get_labels_name(),
                )
            }
        )

    def _model_eval(self, model_fname: str, on_validation: bool = False, complete_path: bool = False) -> None:
        eval_loss = 0.0
        evalloader = self.dataset.get_valloader() if on_validation else self.dataset.get_testloader()
        device = self.config.get_device()
        test_fold = self.dataset.get_fold()
        model_path = model_fname if complete_path else self.experiment.get_model_path(file_name=model_fname)

        # TODO: Improve the logging
        if not on_validation:
            print("Loading model {}".format(model_path))
        # Load the model
        model = self.config.get_model()
        model.to(device=device)
        model.load_state_dict(self.experiment.get_model_state(model_path))
        model.eval()

        criterion = nn.CrossEntropyLoss()

        macro_output = []
        macro_label = []

        # Disable autograd
        with torch.no_grad():
            for batch_id, (data, labels) in enumerate(evalloader):
                # Move to gpu
                data = data.float().to(device)

                labels = labels.to(device)
                labels -= self.dataset.get_labels().min()

                output = model(data)
                loss = criterion(output, labels)
                eval_loss += loss.item()

                # Convert from one-hot to indexing
                output = torch.argmax(output, dim=1)

                # Return to the normalized labels
                output += self.dataset.get_labels().min()
                labels += self.dataset.get_labels().min()

                # Moves the report variables into the cpu to not overload the gpu
                macro_output.append(output.cpu().numpy())
                macro_label.append(labels.cpu().numpy())

        eval_loss = eval_loss / (batch_id + 1)

        if not on_validation:
            self._eval_report(np.concatenate(macro_output), np.concatenate(macro_label), test_fold)

        return eval_loss

    def _classifier_eval(self, on_validation: bool = False) -> None:
        evalloader = self.dataset.get_valloader() if on_validation else self.dataset.get_testloader()
        test_fold = self.dataset.get_fold()

        for data, labels in evalloader:
            data = data.float()
            labels -= self.dataset.get_labels().min()
            output = self.classifier.predict(data)

            # Return to the normalized labels
            output += self.dataset.get_labels().min()
            labels += self.dataset.get_labels().min()

        if not on_validation:
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            if isinstance(output, torch.Tensor):
                output = output.numpy()
            data = {"predicted": output, "real_label": labels}
            self._eval_report(output, labels, test_fold)

        loss = 1 - balanced_accuracy_score(labels, output)
        return loss

    def eval(
        self,
        model_fname: str,
        on_validation: bool = False,
        complete_path: bool = False,
        **kwargs,
    ) -> None:
        if not on_validation:
            print("Testing".center(30, "="))

        if "classifier" in self.config.config:
            return self._classifier_eval(on_validation=on_validation, **kwargs)
        else:
            return self._model_eval(model_fname, on_validation, complete_path)

    def grid_search_train(self) -> Tuple[int, Dict[str, Any]]:
        params_grid = self.config["params_grid"]
        keys = list(params_grid.keys())
        values = list(params_grid.values())

        optimal_val_loss = sys.maxsize
        optimal_epoch = 0
        optimal_params_set = {k: None for k in keys}

        all_combinations = list(itertools.product(*values))
        num_combinations = len(all_combinations)
        for i, combination_values in enumerate(all_combinations):
            # Organize the combination values based on keys
            combination = {key: value for key, value in zip(keys, combination_values)}
            # Log the combination used
            print(" Grid search {}/{} ".format(i + 1, num_combinations).center(30, "="))
            print("Params:\n" + "\n".join([f"{key} : {value}" for key, value in zip(keys, combination_values)]))
            # Train the model overriding these params
            epoch, val_loss = self.train(on_validation=True, **combination)
            if val_loss < optimal_val_loss:
                optimal_epoch = epoch
                optimal_val_loss = val_loss
                optimal_params_set = combination

        # Print the best parameters found
        print()
        print("Best set found".center(30, "-"))
        print("Epoch: {}".format(optimal_epoch))
        print("Validation loss: {}".format(optimal_val_loss))
        print("Params:\n" + "\n".join([f"{key} : {value}" for key, value in optimal_params_set.items()]))
        wandb.summary["{}_best_val_loss".format(self.dataset.get_fold())] = optimal_val_loss

        return optimal_epoch, optimal_params_set

    def finish(self):
        # Save the predictions
        predictions_path = os.path.join(self.experiment.get_results_dirpath(), "predictions.csv")
        predictions = pd.DataFrame(self.data_report)
        predictions.to_csv(predictions_path, index=False)

        # Save predictions in wandb
        wandb.save(predictions_path, policy="now")

        wandb.run.summary["total_balanced_accuracy"] = (
            balanced_accuracy_score(predictions["real_label"], predictions["predicted"]) * 100
        )
        wandb.run.summary["total_f1_macro"] = (
            f1_score(predictions["real_label"], predictions["predicted"], average="macro") * 100
        )
        wandb.run.summary["total_accuracy"] = accuracy_score(predictions["real_label"], predictions["predicted"]) * 100

        y_true_ = predictions["real_label"] - min(predictions["real_label"])
        y_pred_ = predictions["predicted"] - min(predictions["predicted"])

        wandb.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true_,
                    preds=y_pred_,
                    class_names=self.dataset.get_labels_name(),
                )
            }
        )

    @staticmethod
    def set_deterministic(seed: int):
        # Fix seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        np.random.seed(seed)
        random.seed(seed)

        # CUDA convolution determinism
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        # Set cubLAS enviroment variable to guarantee a deterministc behaviour in multiple streams
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
