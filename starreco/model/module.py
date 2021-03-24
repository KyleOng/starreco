import pytorch_lightning as pl
import torch

class Module(pl.LightningModule):
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
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # Convert sparse tensor into dense
        try:
            x = x.to_dense()
            y = y.to_dense()
        except:
            pass

        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Convert sparse tensor into dense
        try:
            x = x.to_dense()
            y = y.to_dense()
        except:
            pass
            
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        # Convert sparse tensor into dense
        try:
            x = x.to_dense()
            y = y.to_dense()
        except:
            pass

        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)