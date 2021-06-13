import argparse
import time
import sys
sys.path.insert(0,"..")

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from starreco.modules import *
from starreco.data import *

exp_id = str(int(time.time()))

# data module
data_module = StarDataModule(batch_size = 1024, 
                                train_val_test_split = [70, 30, 0],
                                num_workers = 0)
data_module.setup()

def objective(trial: optuna.trial.Trial):
    # hyperparameters 
    field_dims = [data_module.dataset.rating.num_users, data_module.dataset.rating.num_items]
    embed_dim = 8
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    # module
    module = MF(field_dims, embed_dim, lr, weight_decay)

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/mf_{exp_id}/version_{trial.number}",
                                          monitor = "val_loss",
                                          filename = "mf-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}")
    prunning_callback = PyTorchLightningPruningCallback(trial, monitor = "val_loss")

    # logger
    logger = TensorBoardLogger("training_logs", name = f"mf_{exp_id}")

    # trainer
    trainer = Trainer(logger = logger,
                      gpus = -1 if torch.cuda.is_available() else None, 
                      max_epochs = 10, 
                      progress_bar_refresh_rate = 10,
                      callbacks = [prunning_callback, checkpoint_callback])
    trainer.fit(module, data_module)

    return trainer.callback_metrics["val_loss"].item()

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description = "Matrix Factorization.")
    parser.add_argument("-p","--pruning",
                        action = "store_true",
                        help = "Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.",)
    args = parser.parse_args()

    # Pruner
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    start_time = time.time()

    # Study
    study = optuna.create_study(direction = "minimize", pruner = pruner)
    study.optimize(objective, n_trials = 100, timeout = 600)

    print("Number of finished trials: {}".format(len(study.trials)))

    # Best trial
    trial = study.best_trial

    print("Value: {}".format(trial.value))
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(f"Time completion: {str('%.2f' %  (time.time() - start_time))} seconds")