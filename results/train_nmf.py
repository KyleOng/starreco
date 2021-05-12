import itertools
import os
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

# load models with the best result among GMF and NCF
gmf_checkpoints = [os.path.join(path, name)
                   for path, subdirs, files in os.walk("checkpoints/gmf")
                   for name in files]
gmf_val_losses = [float(checkpoint.split("=")[-1][:-5]) for checkpoint in gmf_checkpoints]
gmf_best_checkpoint = gmf_checkpoints[gmf_val_losses.index(min(gmf_val_losses))]
gmf = GMF.load_from_checkpoint(gmf_best_checkpoint)

ncf_checkpoints = [os.path.join(path, name)
                   for path, subdirs, files in os.walk("checkpoints/ncf")
                   for name in files]
ncf_val_losses = [float(checkpoint.split("=")[-1][:-5]) for checkpoint in ncf_checkpoints]
ncf_best_checkpoint = ncf_checkpoints[ncf_val_losses.index(min(ncf_val_losses))]
ncf = NCF.load_from_checkpoint(ncf_best_checkpoint)

# all possible hyperparameters combinations
hparams_grid = {
    "gmf_hparams": [gmf.hparams],
    "ncf_hparams": [ncf.hparams],
    "gmf_params": [None, gmf.state_dict()],
    "ncf_params":[None, ncf.state_dict()],
    "freeze_pretrain": [True, False],
}
keys, values = zip(*hparams_grid.items())
hparams_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

# train all possible hyperparameter combinations
current_version = max(0, len(list(os.walk("checkpoints/nmf")))-1)
for i, hparams in enumerate(hparams_grid):
    # module
    module = NMF(**hparams)

    # setup
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/nmf/version_{i + current_version}",
                                          monitor = "val_loss_",
                                          filename = "nmf-{epoch:02d}-{val_loss_:.4f}")
    # logger
    logger = TensorBoardLogger("training_logs", 
                               name = "nmf", 
                               log_graph = True)
    # trainer
    trainer = Trainer(logger = logger,
                      gpus = -1 if torch.cuda.is_available() else None, 
                      max_epochs = 100, 
                      progress_bar_refresh_rate = 2,
                      callbacks=[checkpoint_callback])
    trainer.fit(module, data_module)

    # evaluate
    module_test = NMF.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(module_test, datamodule = data_module)