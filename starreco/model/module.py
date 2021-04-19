import pytorch_lightning as pl
import torch
import time

class BaseModule(pl.LightningModule):
    def __init__(self, lr, weight_decay, criterion):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = self.lr, 
                                     weight_decay = self.weight_decay)
       
        return optimizer
    
    def forward(self, x):
        return self.model(x)

    def evaluate(self, x, y):
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Convert sparse tensor into dense
        x = x.to_dense() if x.layout == torch.sparse_coo else x
        y = y.to_dense() if y.layout == torch.sparse_coo else y
        
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        
        loss = self.evaluate(x, y)
    
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Convert sparse tensor into dense
        x = x.to_dense() if x.layout == torch.sparse_coo else x
        y = y.to_dense() if y.layout == torch.sparse_coo else y
            
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        loss = self.evaluate(x, y)
        
        self.log("val_loss", loss, on_step = True, on_epoch = True, prog_bar = True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        # Convert sparse tensor into dense
        x = x.to_dense() if x.layout == torch.sparse_coo else x
        y = y.to_dense() if y.layout == torch.sparse_coo else y

        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        loss = torch.sqrt(self.criterion(y_hat, y))
        
        self.log("test_loss", loss, prog_bar = True)

    def get_progress_bar_dict(self):
        # Don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)

        if "train_loss" in self.trainer.callback_metrics:
            train_loss = self.trainer.callback_metrics["train_loss"].item()
            items["train_loss"] = train_loss

        if "val_loss" in self.trainer.callback_metrics:
            val_loss = self.trainer.callback_metrics["val_loss"].item()
            items["val_loss"] = val_loss

        return items