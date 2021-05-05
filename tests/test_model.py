import argparse
import sys
sys.path.insert(0,"..")

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from starreco.model import *
from starreco.data import *

model = "mf"
gpu = False
epoch = 5

def quick_test(dm, module):
    logger = TensorBoardLogger("training_logs", name = model, log_graph = True)
    trainer = pl.Trainer(logger=logger,
                         gpus = int(gpu), 
                         max_epochs = epoch, 
                         progress_bar_refresh_rate = 2)
    trainer.fit(module, dm)
    trainer.test(module, datamodule = dm)

    return module


def test_mf():
    dm = StarDataModule(download = "ml-1m", 
                        batch_size = 1024)
    dm.setup()
    mf = MF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, mf)


def test_gmf():
    dm = StarDataModule(download = "ml-1m", 
                        batch_size = 1024)
    dm.setup()
    gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, gmf)


def test_ncf():
    dm = StarDataModule(download = "ml-1m", 
                        batch_size = 1024)
    dm.setup()
    ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, ncf)


def test_nmf(pretrain = False, freeze = True):
    dm = StarDataModule(download = "ml-1m", 
                        batch_size = 1024)
    dm.setup()
    if pretrain:
        gmf = test_gmf()
        ncf = test_ncf()
    else:
        gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    nmf = NMF(gmf.model, ncf.model, freeze_pretrain = freeze)
    return quick_test(dm, nmf)


def test_mdacf():
    dm = StarDataModule(download = "ml-1m", 
                        batch_size = 1024)
    dm.setup()
    mdacf = MDACF(torch.Tensor(dm.user.toarray()), torch.Tensor(dm.item.toarray()))
    return quick_test(dm, mdacf)


def test_sdae(matrix_transpose = False, features_join = False):
    dm = StarDataModule(download = "ml-1m",  
                        features_join = features_join,
                        matrix_transform = True,
                        matrix_transpose = matrix_transpose)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    feature_dim = 0
    if features_join:
        if matrix_transpose:
            feature_dim = dm.item.shape[1]
        else:
            feature_dim = dm.user.shape[1]
    sdae = SDAE(input_output_dim, feature_dim)
    return quick_test(dm, sdae)


def test_gmfpp():
    dm = StarDataModule(download = "ml-1m", 
                        features_join = True)

    dm.setup()
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [16], e_activations = "selu", e_dropouts = 0.5, d_dropouts = 0.5, latent_dropout = 0.5, batch_norm = False, std = 1, noise_factor = 0.3)
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [16], e_activations = "selu", e_dropouts = 0.5, d_dropouts = 0.5, latent_dropout = 0.5, batch_norm = False, std = 1, noise_factor = 0.3)
    gmfpp = GMFPP(user_ae.model, item_ae.model, [dm.dataset.rating.num_items, dm.dataset.rating.num_users], 16)
    return quick_test(dm, gmfpp)


def test_ncfpp():
    dm = StarDataModule(download = "ml-1m", 
                        batch_size = 1024,
                        features_join = True)

    dm.setup()
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [32], e_activations = "selu", d_activations = "relu", e_dropouts = 0.5, 
                   d_dropouts = 0.5, latent_dropout = 0.5, batch_norm = False, std = 0.01, noise_factor = 0.3)
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [32], e_activations = "selu", d_activations = "relu", e_dropouts = 0.5,
                   d_dropouts = 0.5, latent_dropout = 0.5, batch_norm = False, std = 0.01, noise_factor = 0.3)
    ncfpp = NCFPP(user_ae.model, item_ae.model, [dm.dataset.rating.num_items, dm.dataset.rating.num_users], 16, [128, 64, 32], 
                  batch_norm = False, activations = "selu", weight_decay = 1e-3, alpha = 1e-6, beta = 1e-6)
    return quick_test(dm, ncfpp)


def test_nmfpp(pretrain = False, freeze = True):
    dm = StarDataModule(download = "ml-1m", 
                        features_join = True)
    dm.setup()
    
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [16, 8], e_activations = "selu")
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [16, 8], e_activations = "selu")
    gmfpp = GMFPP(user_ae.model, item_ae.model, [dm.dataset.rating.num_items, dm.dataset.rating.num_users])
    ncfpp = NCFPP(user_ae.model, item_ae.model, [dm.dataset.rating.num_items, dm.dataset.rating.num_users])
    nmfpp = NMFPP(gmfpp.model, ncfpp.model, freeze_pretrain = freeze)
    return quick_test(dm, nmfpp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Testing model")
    parser.add_argument("--model", type = str, default = "mf", help = "model")
    parser.add_argument("--epoch", type = int, default = 5, help = "epoch")
    parser.add_argument("--gpu", type = bool, default = False, help = "gpu")

    args = parser.parse_args()

    model = args.model
    epoch = args.epoch
    gpu = args.gpu

    if model == "mf": test_mf()
    elif model == "gmf": test_gmf()
    elif model == "ncf": test_ncf()
    elif model == "nmf": test_nmf()
    elif model == "nmf_pretrain": test_nmf(True, False)
    elif model == "nmf_freeze_pretrain": test_nmf(True, True)
    elif model == "mdacf": test_mdacf()
    elif model == "sdae": test_sdae()
    elif model == "sdae_transpose": test_sdae(True)
    elif model == "sdae_features": test_sdae(False, True)
    elif model == "sdae_transpose_features": test_sdae(True, True)
    elif model == "gmfpp": test_gmfpp()
    elif model == "ncfpp": test_ncfpp()
    elif model == "nmfpp": test_nmfpp()
        