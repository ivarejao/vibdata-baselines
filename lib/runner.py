import os
import sys
import random
import itertools
from typing import Any, Dict, Tuple

import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from matplotlib import pyplot as plt
import seaborn as sns
import wandb
from lib.config import Config
from lib.sampling import DataSampling
from utils.report import ReportDict
from lib.experiment import Experiment
from lib.models.model import Model

# Test
from lib.models.vgg_models import VGGish
from utils.report import array_info

class ExpRunner:
    def __init__(
        self, data_sampling: DataSampling, config: Config, experiment: Experiment, deterministic: bool = True
    ) -> None:
        self.dataset = data_sampling
        self.config = config
        self.model = None
        self.experiment = experiment
        self.metrics = []
        self.fold_distributions = []
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
        
        X_train, y_train = self.dataset.get_train()

        X_train = VGGish(channels=1).fit_transform(X_train, y_train)
        
        unique_elements, counts = np.unique(y_train, return_counts=True)
        
        vggish_passed = "Passed VGGish"
        print(vggish_passed.center(30, "="))
        
        test_fold = self.dataset.get_fold()

        prefix_log = f"fold:{test_fold}/"
        
        self.classifier.fit(X_train, y_train)
        self.fold_distributions.append(counts)
        wandb.log({"Class Distribution": wandb.Table(data=self.fold_distributions, columns=list(self.dataset.get_labels_name()))})
        
        training_name = "End of Training"
        print(training_name.center(30, "="))
        return (X_train, y_train)

    def eval(self):
        print("Testing".center(30, "="))

        test_fold = self.dataset.get_fold()
        macro_output = []
        macro_label = []

        X_test, y_test = self.dataset.get_test() 

        X_test = VGGish(channels=1).fit_transform(X_test, y_test)
        y_pred = self.classifier.predict(X_test)
        
        data = {"predicted": y_pred, "real_label": y_test}
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
        
        # Compute confusion matrix
        cm = confusion_matrix(data["real_label"], data["predicted"])

        # Normalize the confusion matrix if desired
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=self.dataset.get_labels_name(), yticklabels=self.dataset.get_labels_name())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
        
        self.metrics.append([bal_acc, f1_macro, accuracy])
        
        wandb.log({'Confusion Matrix': wandb.Image(plt),
                   "Metrics": wandb.Table(data=self.metrics, columns=["Balanced Accuracy", "F1 Macro", "Accuracy"])}) 
           
        wandb.run.summary[f"{test_fold}_balanced_accuracy"] = bal_acc
        wandb.run.summary[f"{test_fold}_f1_macro"] = f1_macro
        wandb.run.summary[f"{test_fold}_accuracy"] = accuracy
        
    
    def finish(self):
        # Save the predicitons
        predictions_path = os.path.join(self.experiment.get_results_dirpath(), "predictions.csv")
        predictions = pd.DataFrame(self.data_report)
        predictions.to_csv(predictions_path, index=False)

        bal_acc = balanced_accuracy_score(predictions["real_label"], predictions["predicted"])
        f1_macro = f1_score(predictions["real_label"], predictions["predicted"], average="macro")
        accuracy = accuracy_score(predictions["real_label"], predictions["predicted"])
        
        print("End Summary".center(30, "="))
        class_report = classification_report(
            predictions["real_label"],
            predictions["predicted"],
            target_names=self.dataset.get_labels_name(),
            zero_division=0,
        )
        print(class_report)
        print("\nBalanced Accuracy: {:.3f}%".format(bal_acc))
        print("\nF1 Macro: {:.3f}%".format(f1_macro))
        print("\nAccuracy: {:.3f}%".format(accuracy))
        # Save predictions in wandb
        wandb.save(predictions_path, policy="now")

        # Compute confusion matrix
        cm = confusion_matrix(predictions["real_label"], predictions["predicted"])

        # Normalize the confusion matrix if desired
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=self.dataset.get_labels_name(), yticklabels=self.dataset.get_labels_name())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
       
        wandb.log({'Confusion Matrix Total': wandb.Image(plt),
                    'Balanced Accuracy': float(bal_acc) * 100,
                    'F1 Macro': float(f1_macro) * 100,
                    'Accuracy': float(accuracy) * 100
                })
        
        wandb.run.summary["total_balanced_accuracy"] = (
            bal_acc * 100
        )
        wandb.run.summary["total_f1"] = (
            f1_macro * 100
        )
        wandb.run.summary["total_accuracy"] = (
            accuracy * 100
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
