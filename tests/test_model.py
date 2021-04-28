import argparse
import sys
sys.path.insert(0,"..")

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from starreco.model import *
from starreco.data import *
from starreco.data.utils import *

def quick_test(dm, module, gpu = False):
    trainer = pl.Trainer(gpus = int(gpu), max_epochs = 10, progress_bar_refresh_rate = 50)
    trainer.fit(module, dm)

def test_mf():
    dm = StarDataModule(download = "ml-1m")
    dm.setup()
    mf = MF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    quick_test(dm, mf)

def test_mdacf():
    dm = StarDataModule(download = "ml-1m", batch_size = 1024)
    dm.setup()
    mdacf = MDACF(torch.Tensor(dm.user.toarray()), torch.Tensor(dm.item.toarray()))
    quick_test(dm, mdacf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Testing model")
    parser.add_argument("--model", type = str, default = "mf", help = "model")
    parser.add_argument("--gpu", type = bool, default = "mf", help = "gpu")

    args = parser.parse_args()

    model = args.model

    if model == "mf": test_mf()
    elif model == "mdacf": test_mdacf()
        