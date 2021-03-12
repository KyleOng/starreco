import pytorch_lightning as pl

class Module(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        y_hat = self.forward(x)
        
        loss = self.criterion(y_hat, y)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)