import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# Done
class BaseModule(pl.LightningModule):
    """
    Base Module class.
    
    - lr (float): Learning rate.
    - weight_decay (float): L2 regularization rate.
    - criterion: Criterion or objective or loss function.
    """

    def __init__(self, lr, weight_decay, criterion):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        return optimizer

    def backward_loss(self, *batch):
        """
        Calculate loss for backward propagation during training.
        """
        xs = batch[:-1]
        y = batch[-1]
        y_hat = self.forward(*xs)
        loss = self.criterion(y_hat, y)

        return loss

    def logger_loss(self, *batch):
        """
        Calculate loss for logger and evaluation. 
        """
        xs = batch[:-1]
        y = batch[-1]
        y_hat = self.forward(*xs)
        loss = self.criterion(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        batch = [tensor.view(tensor.shape[0], -1) for tensor in batch]
        backward_loss = self.backward_loss(*batch)
        #self.log("backward_loss", backward_loss, on_step = False, on_epoch = True, prog_bar = True)

        logger_loss = self.logger_loss(*batch)
        self.log("train_loss", logger_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        
        return backward_loss

    def validation_step(self, batch, batch_idx):
        batch = [tensor.view(tensor.shape[0], -1) for tensor in batch]
        # Log graph on the first validation step (took less time than training step)
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.log_graph(self, batch[:-1])

        logger_loss = self.logger_loss(*batch)
        self.log("val_loss", logger_loss, on_epoch = True, prog_bar = True, logger = True)

    def test_step(self, batch, batch_idx):
        batch = [tensor.view(tensor.shape[0], -1) for tensor in batch]
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