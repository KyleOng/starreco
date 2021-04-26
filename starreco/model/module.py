import torch
import time
import pytorch_lightning as pl


class BaseModule(pl.LightningModule):
    def __init__(self, 
                 model, 
                 lr, 
                 weight_decay, 
                 criterion):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        return optimizer
        
    def _transform(self, tensor):
        """
        Transform tensor.

        Warning: This method should not be used directly.
        """
        # Convert tensor to dense if tensor is sparse
        if tensor.layout == torch.sparse_coo:
            tensor = tensor.to_dense()

        # Reshape tensor
        return tensor.view(tensor.shape[0], -1)

    def backward_loss(self, *batch):
        """
        Calculate loss for backward propagation.
        """
        y_hat = self.model.forward(*batch[:-1])
        loss = self.criterion(y_hat, batch[-1])

        return loss

    def logger_loss(self, *batch):
        """
        Calculate loss for logger and evaluation. 
        """
        y_hat = self.model.forward(*batch[:-1])
        loss = self.criterion(y_hat, batch[-1])

        return loss

    def training_step(self, batch, batch_idx):
        batch = [self._transform(tensor) for tensor in batch]

        backward_loss = self.backward_loss(*batch)
        self.log("train_loss", backward_loss, on_step = False, on_epoch = True, prog_bar = True)

        logger_loss = self.logger_loss(*batch)
        self.log("train_loss_", logger_loss, on_step = False, on_epoch = True, prog_bar = True)

        return backward_loss

    def validation_step(self, batch, batch_idx):
        batch = [self._transform(tensor) for tensor in batch]

        backward_loss = self.backward_loss(*batch)
        self.log("val_loss", backward_loss, prog_bar = True)

        logger_loss = self.logger_loss(*batch)
        self.log("val_loss_", logger_loss, prog_bar = True)

        return backward_loss

    def test_step(self, batch, batch_idx):
        batch = [self._transform(tensor) for tensor in batch]
        
        logger_loss = self.logger_loss(*batch)
        self.log("test_loss", logger_loss)

    def get_progress_bar_dict(self):
        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)

        """
        if "train_loss" in self.trainer.callback_metrics:
            train_loss = self.trainer.callback_metrics["train_loss"].item()
            items["train_loss"] = train_loss

        if "val_loss" in self.trainer.callback_metrics:
            val_loss = self.trainer.callback_metrics["val_loss"].item()
            items["val_loss"] = val_loss
        """

        return items