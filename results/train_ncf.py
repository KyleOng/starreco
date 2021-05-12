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

# all possible hyperparameters combinations
hparams_grid = {
    "field_dims": [[data_module.dataset.rating.num_users, 
                    data_module.dataset.rating.num_items]],
    "embed_dim": [8, 16, 32, 64],
    "activations": ["relu", "elu", "selu"],
    "dropouts": [0, 0.5],
    "batch_norm": [True, False],
}
keys, values = zip(*hparams_grid.items())
hparams_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

# train all possible hyperparameter combinations
current_version = max(0, len(list(os.walk("checkpoints/ncf")))-1)
for i, hparams in enumerate(hparams_grid):
    # module
    module = NCF(**hparams)

    # setup
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/ncf/version_{i + current_version}",
                                          monitor = "val_loss_",
                                          filename = "ncf-{epoch:02d}-{val_loss_:.4f}")
    # logger
    logger = TensorBoardLogger("training_logs", 
                               name = "ncf", 
                               log_graph = True)
    # trainer
    trainer = Trainer(logger = logger,
                      gpus = -1 if torch.cuda.is_available() else None, 
                      max_epochs = 100, 
                      progress_bar_refresh_rate = 2,
                      callbacks=[checkpoint_callback])
    trainer.fit(module, data_module)

    # evaluate
    module_test = NCF.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(module_test, datamodule = data_module)