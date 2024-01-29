from typing import Any, Dict, Optional, Type

import lightning as L
from torch import nn
from torch.optim import Optimizer
from torchmetrics.classification import Accuracy, F1Score


class VibnetModule(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: Type[Optimizer],
        optimizer_params: Dict[str, Any],
        num_classes: int,
    ):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, input):
        return self.network(input)

    def training_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y = batch
        y = y.reshape(-1)
        z = self.network(x)

        loss = self.loss_fn(z, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        x, y = batch
        y = y.reshape(-1)
        z = self.network(x)

        self.val_acc(z, y)
        self.val_f1(z, y)

        self.log("val/acc", self.val_acc, on_epoch=True, on_step=False)
        self.log("val/f1", self.val_f1, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer_class(self.network.parameters(), **self.optimizer_params)
