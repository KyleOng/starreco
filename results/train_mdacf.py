import itertools
import sys
sys.path.insert(0,"..")

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from starreco.model import *
from starreco.data import *


# data module
data_module = StarDataModule(batch_size = 1024)
data_module.setup()
user = torch.Tensor(data_module.user.toarray())
item = torch.Tensor(data_module.item.toarray())

# all possible hyperparameters combinations
hparams_grid = {
    "user": [user],
    "item": [item],
    "embed_dim": [8, 16, 32, 64],
    #"corrupt_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #"alpha": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    #"beta":[0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
    #"lambda": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}
keys, values = zip(*hparams_grid.items())
hparams_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

# train all possible hyperparameter combinations
current_version = max(0, len(list(os.walk("checkpoints/mdacf")))-1)
for i, hparams in enumerate(hparams_grid):
    # module
    module = MDACF(**hparams)

    # setup
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/mdacf/version_{i + current_version}",
                                          monitor = "val_loss_",
                                          filename = "mdacf-{epoch:02d}-{val_loss_:.4f}")
    # logger
    logger = TensorBoardLogger("training_logs", 
                               name = "mdacf", 
                               log_graph = True)
    # trainer
    trainer = Trainer(logger = logger,
                      gpus = -1 if torch.cuda.is_available() else None, 
                      max_epochs = 100, 
                      progress_bar_refresh_rate = 2,
                      callbacks=[checkpoint_callback])
    trainer.fit(module, data_module)

    # evaluate
    #module_test = MDACF.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(module, datamodule = data_module)