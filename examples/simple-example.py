import os
import sys
sys.path.insert(0,"..")

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from starreco.modules import *
from starreco.data import *
    
# data module
data_module = StarDataModule("ml-1m")
data_module.setup()
    
# module
module = MF([data_module.dataset.rating.num_users, data_module.dataset.rating.num_items])

# setup
# checkpoint callback
current_version = max(0, len(list(os.walk("checkpoints/mf")))-1)
checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/mf/version_{current_version}",
                                      monitor = "val_loss",
                                      filename = "mf-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}")
# logger
logger = TensorBoardLogger("training_logs", name = "mf")
# trainer
trainer = Trainer(logger = logger,
                  gpus = -1 if torch.cuda.is_available() else None, 
                  max_epochs = 100, 
                  progress_bar_refresh_rate = 2,
                  callbacks=[checkpoint_callback])
trainer.fit(module, data_module)

# evaluate
module_test = MF.load_from_checkpoint(checkpoint_callback.best_model_path)
trainer.test(module_test, datamodule = data_module)