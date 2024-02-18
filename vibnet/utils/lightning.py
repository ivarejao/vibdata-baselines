from typing import Any, Dict, Type, Optional

import lightning as L
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torchmetrics.classification import F1Score, Accuracy

class VibnetModule(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: Type[Optimizer],
        optimizer_params: Dict[str, Any],
        num_classes: int,
        **kwargs
    ):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if "lr_scheduler_class" in kwargs:
            self.lr_scheduler_class : Type[LRScheduler] = kwargs["lr_scheduler_class"]
            self.lr_scheduler_params :  Dict[str, Any] = kwargs["lr_scheduler_params"]

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, input):
        return self.network(input)

    def training_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y = batch
        y = y.reshape(-1)
        z = self.network(x)

        loss = self.loss_fn(z, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        y = y.reshape(-1)
        z = self.network(x)

        self.val_acc(z, y)
        self.val_f1(z, y)

        self.log("val/acc", self.val_acc, on_epoch=True, on_step=False)
        self.log("val/f1", self.val_f1, on_epoch=True, on_step=False)

        if hasattr(self, "lr_scheduler_class"):
            self.log("train/lr", self.lr_schedulers().get_last_lr()[0], on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch)

    def configure_optimizers(self) -> Dict[str, Optimizer | LRScheduler]:
        optimizer = self.optimizer_class(self.network.parameters(), **self.optimizer_params)
        ret = {"optimizer": optimizer}
        if hasattr(self, "lr_scheduler_class"):
            lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
            ret["lr_scheduler"] = lr_scheduler
        return ret
    