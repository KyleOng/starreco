import argparse
import pdb
import itertools
import time
import sys
sys.path.insert(0,"..")

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from starreco.modules import *
from starreco.data import *
from starreco.evaluation import *

dm = None

# Testing
def quick_test(dm, module):
    global gpu
    logger = TensorBoardLogger("training_logs", name = module_name, log_graph = True)
    trainer = pl.Trainer(logger = logger,
                         gpus = -1 if gpu else None, 
                         max_epochs = 1, 
                         progress_bar_refresh_rate = 2,
                         #limit_train_batches = 0.1,
                         #limit_val_batches = 0.1,
                         #limit_test_batches = 0.1,
                         weights_summary = "full")
    trainer.fit(module, dm)
    trainer.test(module, datamodule = dm)
    return module

# Testing
def test_mf():
    dm = StarDataModule()
    dm.setup()
    mf = MF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, mf)

# Testing
def test_gmf():
    dm = StarDataModule()
    dm.setup()
    gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, gmf)

# Testing
def test_ncf():
    dm = StarDataModule()
    dm.setup()
    ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, ncf)

# Testing
def test_nmf(shared_embed = None, freeze = True):
    dm = StarDataModule()
    dm.setup()

    gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    nmf = NMF(gmf.hparams, ncf.hparams, shared_embed, weight_decay = 0)
    nmf.load_pretrain_weights(gmf.state_dict(), ncf.state_dict(), freeze)
    
    nmf = quick_test(dm, nmf)
    pdb.set_trace()
    return nmf

# Testing
def test_mdacf():
    dm = StarDataModule()
    dm.setup()
    mdacf = MDACF(torch.Tensor(dm.user.toarray()), torch.Tensor(dm.item.toarray()))
    return quick_test(dm, mdacf)

# Done
def test_gmfpp(load = False):
    dm = StarDataModule(add_features = True, user_features_ignore=["zipCode"], item_features_ignore = ["plot"])
    dm.setup()
    user_ae_hparams = dict(input_output_dim = dm.user_X.shape[-1], hidden_dims = [8])
    item_ae_hparams = dict(input_output_dim = dm.item_X.shape[-1], hidden_dims = [8])
    gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    gmfpp = GMFPP(user_ae_hparams, item_ae_hparams, **gmf.hparams)
    if load:
        gmfpp.load_pretrain_weights(gmf.state_dict())
    gmfpp = quick_test(dm, gmfpp)
    return gmfpp

# Done
def test_ncfpp(load = False):
    dm = StarDataModule(add_features = True, user_features_ignore=["zipCode"], item_features_ignore = ["plot"])
    dm.setup()
    user_ae_hparams = dict(input_output_dim = dm.user_X.shape[-1], hidden_dims = [8])
    item_ae_hparams = dict(input_output_dim = dm.item_X.shape[-1], hidden_dims = [8])
    ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    ncfpp = NCFPP(user_ae_hparams, item_ae_hparams, **ncf.hparams)
    if load:
        ncfpp.load_pretrain_weights(ncf.state_dict())
    ncfpp = quick_test(dm, ncfpp)
    return ncfpp

# Testing
def test_nmfpp(shared_embed = None, shared_sdaes = None, load_all = False, freeze = True):
    global dm
    if dm is None:
        dm = StarDataModule(add_features = True)
        dm.setup()
    user_ae_hparams = dict(input_output_dim = dm.user_X.shape[-1], hidden_dims = [8])
    item_ae_hparams = dict(input_output_dim = dm.item_X.shape[-1], hidden_dims = [8])
    gmf = GMF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    gmfpp = GMFPP(user_ae_hparams, item_ae_hparams, **gmf.hparams)
    ncf = NCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    ncfpp = NCFPP(user_ae_hparams, item_ae_hparams, **ncf.hparams)
    nmf = NMF(gmf.hparams, ncf.hparams, shared_embed.replace("++","") if shared_embed else shared_embed)
    nmfpp = NMFPP(gmfpp.hparams, ncfpp.hparams, shared_embed, shared_sdaes, weight_decay = 0)
    if load_all:
        nmfpp.load_all_pretrain_weights(gmfpp.state_dict(), ncfpp.state_dict(), freeze)
    else:
        nmfpp.load_nmf_pretrain_weights(nmf.state_dict(), freeze)
    nmfpp = quick_test(dm, nmfpp)
    return nmfpp

# Testing
def test_fm():
    dm = StarDataModule()
    dm.setup()
    fm = FM([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, fm)

# Testing
def test_nfm():
    dm = StarDataModule()
    dm.setup()
    nfm = NFM([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, nfm)

# Testing
def test_wdl():
    dm = StarDataModule()
    dm.setup()
    wdl = WDL([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, wdl)

# Testing
def test_dfm():
    dm = StarDataModule()
    dm.setup()
    dfm = DFM([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, dfm)

# Testing
def test_xdfm():
    dm = StarDataModule()
    dm.setup()
    xdfm = XDFM([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, xdfm)
 
# Testing
def test_oncf():
    dm = StarDataModule()
    dm.setup()
    oncf = ONCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items], 64, 64)
    return quick_test(dm, oncf)

# Testing
def test_cnndcf():
    dm = StarDataModule(batch_size = 2048)
    dm.setup()
    cnndcf = CNNDCF([dm.dataset.rating.num_users, dm.dataset.rating.num_items])
    return quick_test(dm, cnndcf)

# Testing
def test_autorec(matrix_transpose = False):
    dm = StarDataModule(matrix_form = True,
                        matrix_transpose = matrix_transpose)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    autorec = AutoRec(input_output_dim)
    return quick_test(dm, autorec)

# Testing
def test_deeprec(matrix_transpose = False):
    dm = StarDataModule(matrix_form = True,
                        matrix_transpose = matrix_transpose)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    deeprec = DeepRec(input_output_dim)
    return quick_test(dm, deeprec)

# Testing
def test_cfn(matrix_transpose = False, extra_input_all = True):
    dm = StarDataModule(matrix_form = True,
                        add_features = True,
                        matrix_transpose = matrix_transpose)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    if matrix_transpose:
        feature_dim = dm.item.shape[1]
    else:
        feature_dim = dm.user.shape[1]
    cfn = CFN(input_output_dim, feature_dim = feature_dim, feature_input_all = extra_input_all)
    return quick_test(dm,  cfn)

# Testing
def test_cdae(matrix_transpose = False):
    dm = StarDataModule(matrix_form = True,
                        matrix_transpose = matrix_transpose,
                        add_ids = True)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    
    cdae = CDAE(input_output_dim)
    return quick_test(dm, cdae)

# Testing
def test_sdaecf(matrix_transpose = False):
    dm = StarDataModule(matrix_form = True,
                        matrix_transpose = matrix_transpose)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    
    sdaecf = SDAECF(input_output_dim)
    return quick_test(dm, sdaecf)

# Testing
def test_ccae(matrix_transpose = False):
    dm = StarDataModule(matrix_form = True,
                        matrix_transpose = matrix_transpose)
    dm.setup()
    input_output_dim = dm.dataset.rating.num_users if matrix_transpose else dm.dataset.rating.num_items
    ccae = CCAE(input_output_dim)
    return quick_test(dm, ccae)

# Testing
def test_cmf(pretrain = True):
    dm = StarDataModule(add_features = True,
                        user_features_ignore = ["zipCode"],
                        item_features_ignore = ["genre"])
    dm.setup()
    user_dim = dm.dataset.rating.num_users
    word_dim = dm.item_preprocessor.column_transformer.named_transformers_["document"].vocab_size + 2
    max_len = dm.item.shape[-1]
    vocab_map = dm.item_preprocessor.column_transformer.named_transformers_["document"].vocab_map

    cmf = CMF(user_dim, word_dim, max_len)
    if pretrain:
        cmf.load_pretrain_embeddings(vocab_map)
    return quick_test(dm, cmf)

# Testing
def test_fgcnn():
    dm = StarDataModule(add_features = True,
                        user_features_ignore = ["zipCode"],
                        item_features_ignore = ["plot"],
                        batch_size = 1024)
    dm.setup()

    user_feature_names = np.concatenate(list(dm.user_preprocessor.get_feature_names().values()))
    user_features_split = {}
    for user_feature_name in user_feature_names:
        column_name = user_feature_name.split("_")[0]
        if column_name in user_features_split:
            user_features_split[column_name] += 1
        else:
            user_features_split[column_name] = 1
    user_features_split = list(user_features_split.values())

    item_feature_names = np.concatenate(list(dm.item_preprocessor.get_feature_names().values()))
    item_features_split = {}
    for item_feature_name in item_feature_names:
        column_name = item_feature_name.split("_")[0]
        if column_name in item_features_split:
            item_features_split[column_name] += 1
        else:
            item_features_split[column_name] = 1
    item_features_split = list(item_features_split.values())

    fgcnn = FGCNN(user_features_split, item_features_split, [dm.dataset.rating.num_users, dm.dataset.rating.num_items])

    return quick_test(dm, fgcnn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Testing module")
    parser.add_argument("--module", type = str, default = "mf", help = "module")
    parser.add_argument("--gpu", type = bool, default = False, help = "gpu")
    #parser.add_argument("--weights_summary", type = str, default = "top", help = "module_name")

    args = parser.parse_args()

    module_name = args.module
    gpu = args.gpu
    #weights_summary = args.weights_summary

    # Testing
    if module_name == "mf": test_mf()
    # Testing
    elif module_name == "gmf": test_gmf()
    # Testing
    elif module_name == "ncf": test_ncf()
    # Testing
    elif module_name == "nmf_0": test_nmf(None, False) 
    elif module_name == "nmf_1": test_nmf(None, True) 
    elif module_name == "nmf_2": test_nmf("gmf", False) 
    elif module_name == "nmf_3": test_nmf("gmf", True) 
    elif module_name == "nmf_4": test_nmf("ncf", False) 
    elif module_name == "nmf_5": test_nmf("ncf", True) 
    # Testing
    elif module_name == "gmfpp_0": test_gmfpp(False)
    elif module_name == "gmfpp_1": test_gmfpp(True)
    # Testing
    elif module_name == "ncfpp_0": test_ncfpp(False)
    elif module_name == "ncfpp_1": test_ncfpp(True)
    # Testing
    elif module_name == "nmfpp":
        hparams_grid = dict(
            shared_embed = [None, "gmf++", "ncf++"], 
            shared_sdaes = [None, "gmf++", "ncf++"], 
            load_all = [True, False], 
            freeze = [True, False]
        )
        keys, values = zip(*hparams_grid.items())
        hparams_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"run {len(hparams_grid)} trials/tests")
        for i, hparams in enumerate(hparams_grid):
            try:
                torch.cuda.empty_cache()
                print(f"BEGIN: trial {i} ", hparams)
                test_nmfpp(**hparams)
            except Exception as e:
                print(f"FAIL: trial {i}", hparams)
                raise(e)
            else:
                print(f"PASS: trial {i} ", hparams)
                time.sleep(5) # load_all, freeze
    elif module_name == "nmfpp_00": test_nmfpp(None, None, False, False)
    elif module_name == "nmfpp_01": test_nmfpp(None, None, False, True)
    elif module_name == "nmfpp_02": test_nmfpp(None, None, True, False)
    elif module_name == "nmfpp_03": test_nmfpp(None, None, True, True)
    elif module_name == "nmfpp_04": test_nmfpp(None, "gmf++", False, False)
    elif module_name == "nmfpp_05": test_nmfpp(None, "gmf++", False, True)
    elif module_name == "nmfpp_06": test_nmfpp(None, "gmf++", True, False)
    elif module_name == "nmfpp_07": test_nmfpp(None, "gmf++", True, True)
    elif module_name == "nmfpp_08": test_nmfpp(None, "ncf++", False, False)
    elif module_name == "nmfpp_09": test_nmfpp(None, "ncf++", False, True)
    elif module_name == "nmfpp_10": test_nmfpp(None, "ncf++", True, False)
    elif module_name == "nmfpp_11": test_nmfpp(None, "ncf++", True, True)
    elif module_name == "nmfpp_12": test_nmfpp("gmf++", None, False, False)
    elif module_name == "nmfpp_13": test_nmfpp("gmf++", None, False, True)
    elif module_name == "nmfpp_14": test_nmfpp("gmf++", None, True, False)
    elif module_name == "nmfpp_15": test_nmfpp("gmf++", None, True, True)
    elif module_name == "nmfpp_16": test_nmfpp("gmf++", "gmf++", False, False)
    elif module_name == "nmfpp_17": test_nmfpp("gmf++", "gmf++", False, True)
    elif module_name == "nmfpp_18": test_nmfpp("gmf++", "gmf++", True, False)
    elif module_name == "nmfpp_19": test_nmfpp("gmf++", "gmf++", True, True)
    elif module_name == "nmfpp_20": test_nmfpp("gmf++", "ncf++", False, False)
    elif module_name == "nmfpp_21": test_nmfpp("gmf++", "ncf++", False, True)
    elif module_name == "nmfpp_22": test_nmfpp("gmf++", "ncf++", True, False)
    elif module_name == "nmfpp_23": test_nmfpp("gmf++", "ncf++", True, True)
    elif module_name == "nmfpp_24": test_nmfpp("ncf++", None, False, False)
    elif module_name == "nmfpp_25": test_nmfpp("ncf++", None, False, True)
    elif module_name == "nmfpp_26": test_nmfpp("ncf++", None, True, False)
    elif module_name == "nmfpp_27": test_nmfpp("ncf++", None, True, True)
    elif module_name == "nmfpp_28": test_nmfpp("ncf++", "gmf++", False, False)
    elif module_name == "nmfpp_29": test_nmfpp("ncf++", "gmf++", False, True)
    elif module_name == "nmfpp_30": test_nmfpp("ncf++", "gmf++", True, False)
    elif module_name == "nmfpp_31": test_nmfpp("ncf++", "gmf++", True, True)
    elif module_name == "nmfpp_32": test_nmfpp("ncf++", "ncf++", False, False)
    elif module_name == "nmfpp_33": test_nmfpp("ncf++", "ncf++", False, True)
    elif module_name == "nmfpp_34": test_nmfpp("ncf++", "ncf++", True, False)
    elif module_name == "nmfpp_35": test_nmfpp("ncf++", "ncf++", True, True)
    # Testing
    elif module_name == "mdacf": test_mdacf()
    # Testing
    elif module_name == "fm": test_fm()
    # Testing
    elif module_name == "nfm": test_nfm()
    # Testing
    elif module_name == "wdl": test_wdl()
    # Testing
    elif module_name == "dfm": test_dfm()
    # Testing
    elif module_name == "xdfm": test_xdfm()
    # Testing
    elif module_name == "oncf": test_oncf()
    # Testing
    elif module_name == "cnndcf": test_cnndcf()
    # Testing
    elif module_name == "autorec_0": test_autorec(False)
    elif module_name == "autorec_1": test_autorec(True)
    # Testing
    elif module_name == "deeprec_0": test_deeprec(False)
    elif module_name == "deeprec_1": test_deeprec(True)
    # Testing
    elif module_name == "cfn_0": test_cfn(False, False)
    elif module_name == "cfn_1": test_cfn(False, True)
    elif module_name == "cfn_2": test_cfn(True, False)
    elif module_name == "cfn_3": test_cfn(True, True)
    # Testing
    elif module_name == "cdae_0": test_cdae(False)
    elif module_name == "cdae_1": test_cdae(True)
    # Testing
    elif module_name == "sdaecf_0": test_sdaecf(False)
    elif module_name == "sdaecf_1": test_sdaecf(True)
    # Testing
    elif module_name == "ccae_0": test_ccae(False)
    elif module_name == "ccae_1": test_ccae(True)
    # Testing
    elif module_name == "cmf_0": test_cmf(False)
    elif module_name == "cmf_1": test_cmf(True)
    # Testing
    elif module_name == "fgcnn": test_fgcnn()

        