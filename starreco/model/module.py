import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# Done
class BaseModule(pl.LightningModule):
    """
    Base Module class.
    
    - lr (float): Learning rate.
    - l2_lambda (float): L2 regularization rate.
    - criterion (F): Criterion or objective or loss function.
    """

    def __init__(self, 
                 lr:float, 
                 l2_lambda:float, 
                 criterion:F):
        super().__init__()
        self.lr = lr
        self.l2_lambda = l2_lambda
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.l2_lambda)
        return optimizer
        
    def _transform(self, tensor:torch.Tensor):
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
        """Calculate loss for backward propagation."""
        xs = batch[:-1]
        y = batch[-1]
        y_hat = self.forward(*xs)
        loss = self.criterion(y_hat, y)

        return loss

    def logger_loss(self, *batch):
        """Calculate loss for logger and evaluation. """
        xs = batch[:-1]
        y = batch[-1]
        y_hat = self.forward(*xs)
        loss = self.criterion(y_hat, y)

        return loss

    def training_step(self, batch:list, batch_idx:int):
        batch = [self._transform(tensor) for tensor in batch]

        backward_loss = self.backward_loss(*batch)
        self.log("train_loss", backward_loss, on_step = False, on_epoch = True, prog_bar = True)

        logger_loss = self.logger_loss(*batch)
        self.log("train_loss_", logger_loss, on_step = False, on_epoch = True, prog_bar = True, logger = True)
        
        return backward_loss

    def validation_step(self, batch:list, batch_idx:int):
        batch = [self._transform(tensor) for tensor in batch]

        # Log graph on the first validation step (took less time than training step)
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.log_graph(self, batch[:-1])

        backward_loss = self.backward_loss(*batch)
        self.log("val_loss", backward_loss, on_epoch = True, prog_bar = True, logger = True)

        logger_loss = self.logger_loss(*batch)
        self.log("val_loss_", logger_loss, on_epoch = True, prog_bar = True, logger = True)

        return backward_loss

    def test_step(self, batch:list, batch_idx:int):
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