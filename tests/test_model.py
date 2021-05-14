import argparse
import pdb
import sys
sys.path.insert(0,"..")

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from starreco.model import *
from starreco.data import *

def quick_test(dm, module):
    logger = TensorBoardLogger("training_logs", name = model, log_graph = True)
    trainer = pl.Trainer(logger=logger,
                         gpus = int(gpu), 
                         max_epochs = epoch, 
                         progress_bar_refresh_rate = 2)
    trainer.fit(module, dm)
    trainer.test(module, datamodule = dm)
    pdb.set_trace()
    return module


def test_mf():
    dm = StarDataModule()
    dm.setup()
    mf = MF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, mf)


def test_gmf():
    dm = StarDataModule()
    dm.setup()
    gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, gmf)


def test_ncf():
    dm = StarDataModule()
    dm.setup()
    ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, ncf)


def test_nmf(pretrain = False, freeze = True):
    dm = StarDataModule()
    dm.setup()
    if pretrain:
        gmf = test_gmf()
        ncf = test_ncf()
        nmf = NMF(gmf.hparams, ncf.hparams, gmf.state_dict(), ncf.state_dict(), freeze_pretrain = freeze)
    else:
        gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        nmf = NMF(gmf.hparams, ncf.hparams, freeze_pretrain = freeze)
    return quick_test(dm, nmf)


def test_mdacf():
    dm = StarDataModule()
    dm.setup()
    mdacf = MDACF(torch.Tensor(dm.user.toarray()), torch.Tensor(dm.item.toarray()))
    return quick_test(dm, mdacf)


def test_sdae(matrix_transpose = False, features_join = False):
    dm = StarDataModule(features_join = features_join,
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


def test_gmfpp(params = False):
    dm = StarDataModule(features_join = True)

    dm.setup()
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [8])
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [8])
    if params:
        gmf = test_gmf()
        gmfpp = GMFPP(user_ae.hparams, item_ae.hparams, gmf.hparams, gmf.state_dict())
    else:
        gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        gmfpp = GMFPP(user_ae.hparams, item_ae.hparams, gmf.hparams)
    
    return quick_test(dm, gmfpp)


def test_ncfpp(params = False):
    dm = StarDataModule(features_join = True)

    dm.setup()
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [8])
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [8])
    if params:
        ncf = test_ncf()
        ncfpp = NCFPP(user_ae.hparams, item_ae.hparams, ncf.hparams, ncf.state_dict())
    else:
        ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        ncfpp = NCFPP(user_ae.hparams, item_ae.hparams, ncf.hparams)
    return quick_test(dm, ncfpp)


def test_nmfpp(params = False):
    dm = StarDataModule(features_join = True)

    dm.setup()
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [8])
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [8])
    if params:
        nmf = test_nmf(True, True)
        nmfpp = NMFPP(user_ae.hparams, item_ae.hparams, nmf.hparams, nmf.state_dict())
    else:
        gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items], hidden_dims = [8, 4])
        nmf = NMF(gmf.hparams, ncf.hparams, gmf.state_dict(), ncf.state_dict(), freeze_pretrain = True)
        nmfpp = NMFPP(user_ae.hparams, item_ae.hparams, nmf.hparams, nmf.state_dict())
    pdb.set_trace()
    return quick_test(dm, nmfpp)


def test_nmfs(pretrain = False, freeze = True, params = False):
    dm = StarDataModule(features_join = True)
    dm.setup()
    user_ae = SDAE(dm.user_X.shape[-1], hidden_dims = [8])
    item_ae = SDAE(dm.item_X.shape[-1], hidden_dims = [8])
    if pretrain:
        gmfpp = test_gmfpp(params)
        ncfpp = test_ncfpp(params)
        nmfs = NMFS(gmfpp.hparams, ncfpp.hparams, gmfpp.state_dict(), ncfpp.state_dict(), freeze_pretrain = freeze)
    else:
        gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        gmfpp = GMFPP(user_ae.hparams, item_ae.hparams, gmf.hparams)
        ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
        ncfpp = NCFPP(user_ae.hparams, item_ae.hparams, ncf.hparams)
        nmfs = NMFS(gmfpp.hparams, ncfpp.hparams)

    return quick_test(dm, nmfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Testing model")
    parser.add_argument("--model", type = str, default = "mf", help = "model")
    parser.add_argument("--epoch", type = int, default = 2, help = "epoch")
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
    elif model == "gmfpp_params": test_gmfpp(True)
    elif model == "ncfpp": test_ncfpp()
    elif model == "ncfpp_params": test_ncfpp(True)
    elif model == "nmfpp": test_nmfpp()
    elif model == "nmfpp_params": test_nmfpp(True)
    elif model == "nmfs": test_nmfs()
    elif model == "nmfs_pretrain": test_nmfs(True, False)
    elif model == "nmfs_pretrain_params": test_nmfs(True, False, True)
    elif model == "nmfs_freeze_pretrain": test_nmfs(True, True)
    elif model == "nmfs_freeze_pretrain_params": test_nmfs(True, True, True)
        