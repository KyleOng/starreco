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
data_module = StarDataModule(batch_size = 1024,
                             features_join = True)
data_module.setup()

# load gmf with the best result
gmf_checkpoints = [os.path.join(path, name)
                   for path, subdirs, files in os.walk("checkpoints/gmf")
                   for name in files]
gmf_val_losses = [float(checkpoint.split("=")[-1][:-5]) for checkpoint in gmf_checkpoints]
gmf_best_checkpoint = gmf_checkpoints[gmf_val_losses.index(min(gmf_val_losses))]
gmf = GMF.load_from_checkpoint(gmf_best_checkpoint)

# all possible hyperparameters combinations
ae_hparams_grid = {
    "hidden_dims": [[8],[16],[16,8]],
    "e_activations": ["relu", "elu", "selu"],
    "dropouts": [0, 0.5],
    "batch_norm": [False],

}

hparams_grid = {
    "user_ae_hparams": [dict(input_output_dim = data_module.user_X.shape[-1], 
                             hidden_dims = [16],
                             std = 0.01)],
    "item_ae_hparams": [dict(input_output_dim = data_module.item_X.shape[-1], 
                             hidden_dims = [16],
                             std = 0.01)],
    "gmf_hparams": [gmf.hparams],
    "gmf_params": [gmf.state_dict()],
    "alpha": [1],
    "beta": [1]
}
keys, values = zip(*hparams_grid.items())
hparams_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

# train all possible hyperparameter combinations
current_version = max(0, len(list(os.walk("checkpoints/gmfpp")))-1)
for i, hparams in enumerate(hparams_grid):
    # module
    module = GMFPP(**hparams)

    # setup
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/gmfpp/version_{i + current_version}",
                                          monitor = "val_loss_",
                                          filename = "gmfpp-{epoch:02d}-{val_loss_:.4f}")
    # logger
    logger = TensorBoardLogger("training_logs", 
                               name = "gmfpp", 
                               log_graph = True)
    # trainer
    trainer = Trainer(logger = logger,
                      gpus = -1 if torch.cuda.is_available() else None, 
                      max_epochs = 30, 
                      progress_bar_refresh_rate = 2,
                      callbacks=[checkpoint_callback])
    trainer.fit(module, data_module)

    # evaluate
    module_test = GMFPP.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(module_test, datamodule = data_module)