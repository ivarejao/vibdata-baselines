import os
import sys
import random
import itertools
from typing import Any, Dict, Tuple

import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

import wandb
from lib.config import Config
from lib.sampling import DataSampling
from utils.report import ReportDict
from lib.experiment import Experiment
from lib.models.model import Model


class ExpRunner:
    def __init__(
        self, data_sampling: DataSampling, config: Config, experiment: Experiment, deterministic: bool = True
    ) -> None:
        self.dataset = data_sampling
        self.config = config
        self.model = None
        self.experiment = experiment
        self.data_report = ReportDict(["real_label", "predicted", "fold", "dataset"])
        self.classifier = self.create_rf(self.config["seed"], -2)
        
        if deterministic:
            self.set_deterministic(self.config["seed"])
            
    
    def __create_validation_cv(self, random_state=None,
                         test_size: float = 1 / 9) -> StratifiedShuffleSplit:
        """Creates default cv splitter for validation

        Args:
            random_state: seed for splitting
            test_size: Test size

        Returns:
            StratifiedShuffleSplit cv
        """
        cv = StratifiedShuffleSplit(n_splits=1,
                                    test_size=test_size,
                                    random_state=random_state)
        return cv
    
    def create_rf(self, random_state=None, n_jobs = -2, val_size = 1 / 9):
        """Create random forest with grid search

        Args:
            random_state: Seed to use with validation split and algorithm
            n_jobs: Number of process to execute the model.
            val_size: Validation size

        Returns:
            GridSearchCV object with random forest.
        """
        rf = RandomForestClassifier(n_estimators=1000,
                                    random_state=random_state,
                                    n_jobs=n_jobs)
        cv = self.__create_validation_cv(random_state, test_size=val_size)
        params = {'max_features': [2, 3, 4, 5]}
        gscv = GridSearchCV(rf, params, cv=cv, scoring='f1_macro')
        return gscv

    def train(self):
        training_name = "Training"
        print(training_name.center(30, "="))

        device = self.config.get_device()
        model = self.config.get_model()
        model.to(device)
        model.apply(Model.reset_weights)
        
        train_loader = self.dataset.get_trainloader()
        test_fold = self.dataset.get_fold()

        prefix_log = f"fold:{test_fold}/"
        
        X = []
        y = []
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            # Normalize the labels
            labels -= self.dataset.get_labels().min()
            with autocast(device_type="cuda", dtype=torch.float16):
                # Perform forward pass
                outputs = model(inputs)
                X.append(outputs.detach().cpu().numpy())
                y.append(labels.detach().cpu().numpy())

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        self.classifier.fit(X, y)
        
        training_name = "End of Training"
        print(training_name.center(30, "="))
        return (X, y)

    def eval(self):
        print("Testing".center(30, "="))

        test_loader = self.dataset.get_testloader()
        device = self.config.get_device()
        test_fold = self.dataset.get_fold()
        
        model = self.config.get_model()
        model.to(device=device)
        model.eval()

        macro_output = []
        macro_label = []

        # Disable autograd
        with torch.no_grad():
            for batch_id, (data, labels) in enumerate(test_loader):
                # Move to gpu
                data = data.float().to(device)
                labels = labels.to(device)
                # Normalize labels
                labels -= self.dataset.get_labels().min()

                output = model(data)
                
                output = self.classifier.predict(output.detach().cpu().numpy())
                labels += self.dataset.get_labels().min()

                # Moves the report variables into the cpu to not overload the gpu
                macro_output.append(output)
                macro_label.append(labels.cpu().numpy())

        data = {"predicted": np.concatenate(macro_output), "real_label": np.concatenate(macro_label)}
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
        f1_macro = f1_score(data["real_label"], data["predicted"], average='macro') * 100
        accuracy = accuracy_score(data["real_label"], data["predicted"]) * 100
        
        print("\nBalanced Accuracy: {:.3f}%".format(bal_acc))
        print("\nF1 Macro: {:.3f}%".format(f1_macro))
        print("\nAccuracy: {:.3f}%".format(accuracy))
        wandb.run.summary[f"{test_fold}_balanced_accuracy"] = bal_acc
        wandb.run.summary[f"{test_fold}_f1_macro"] = f1_macro
        wandb.run.summary[f"{test_fold}_accuracy"] = accuracy
        

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
        # Save the predicitons
        predictions_path = os.path.join(self.experiment.get_results_dirpath(), "predictions.csv")
        predicitons = pd.DataFrame(self.data_report)
        predicitons.to_csv(predictions_path, index=False)

        # Save predictions in wandb
        wandb.save(predictions_path, policy="now")

        wandb.run.summary["total_balanced_accuracy"] = (
            balanced_accuracy_score(predicitons["real_label"], predicitons["predicted"]) * 100
        )
        wandb.run.summary["total_f1"] = (
            f1_score(predicitons["real_label"], predicitons["predicted"], average="macro") * 100
        )
        wandb.run.summary["total_accuracy"] = (
            accuracy_score(predicitons["real_label"], predicitons["predicted"]) * 100
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
