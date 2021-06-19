import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MyModelA(pl.LightningModule):
    def __init__(self, hidden_dim = 10):
        super(MyModelA, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, 2)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
        
    def forward(self, x):
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        return F.mse_loss(self.forward(x), y)
    
class MyModelB(pl.LightningModule):
    def __init__(self, hidden_dim = 10):
        super(MyModelB, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, 2)
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
        
    def forward(self, x):
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x,y = batch
        return F.mse_loss(self.forward(x), y)

class MyEnsemble(pl.LightningModule):
    def __init__(self, 
                 modelA_hparams, modelB_hparams,
                 modelA_params = None, modelB_params = None):
        super(MyEnsemble, self).__init__()
        self.modelA = MyModelA(**modelA_hparams)
        self.modelB = MyModelB(**modelB_hparams)

        if modelA_params:
            self.modelA.load_state_dict(modelA_params)
        if modelB_params:
            self.modelB.load_state_dict(modelB_params)

        self.modelA.freeze()
        self.modelB.freeze()
        self.classifier = torch.nn.Linear(4, 2)

        self.save_hyperparameters(ignore = ["modelA_params", "modelB_params"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.forward(x), y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.forward(x), y)


dl = DataLoader(TensorDataset(torch.randn(1000, 10), 
                              torch.randn(1000, 2)), 
                batch_size = 10)

modelA = MyModelA()
modelB = MyModelB()

# pretrained modelA and modelB
trainerA = pl.Trainer(gpus = 0, max_epochs = 5, progress_bar_refresh_rate = 50)
trainerA.fit(modelA, dl)
trainerB = pl.Trainer(gpus = 0, max_epochs = 5, progress_bar_refresh_rate = 50)
trainerB.fit(modelB, dl)

checkpoint_callback = pl.callbacks.ModelCheckpoint()

# modelA and modelB contains pretrained weights
model = MyEnsemble(modelA.hparams, modelB.hparams, modelA.state_dict(), modelB.state_dict())
trainer = pl.Trainer(gpus = 0, max_epochs = 5, progress_bar_refresh_rate = 50, callbacks=[checkpoint_callback])
trainer.fit(model, dl)

# Check all model weights are equal after training
model_test = MyEnsemble.load_from_checkpoint(checkpoint_callback.best_model_path,
                                             hparams_file = checkpoint_callback.best_model_path.split("checkpoints")[0]+"hparams.yaml")

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly!')

compare_models(modelA, model.modelA)
compare_models(model.modelA, model_test.modelA)
compare_models(modelB, model.modelB)
compare_models(model.modelB, model_test.modelB)
