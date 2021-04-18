from typing import Union
import copy

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .gmfpp import GMFPP
from .ncfpp import NCFPP

class NMFPP(torch.nn.Module):
    """
    Neural Matrix Factorizatgion ++
    """
    def __init__(self, gmfpp:GMFPP, ncfpp:NCFPP,
                 pretrain:bool = False):
        super().__init__()
        self.gmfpp = copy.deepcopy(gmfpp)
        self.ncfpp = copy.deepcopy(ncfpp)

        # Freeze embedding weights if pretrain
        if pretrain:
            self.gmfpp.embedding.embedding.weight.requires_grad = False
            self.ncfpp.embedding.embedding.weight.requires_grad = False

        # Remove GMFPP output layer
        self.gmfpp.slp = None

        # Add GMF embedding dim to `input_dim`
        input_dim = self.gmfpp.embedding.embedding.weight.shape[1]

        # Remove NCFPP output layer and freeze hidden weights and biases
        ncfpp_mlp_blocks = []
        for i, module in enumerate(self.ncfpp.mlp.mlp):
            # Freeze weights and bias if pretrain
            if hasattr(self.ncfpp.mlp.mlp[i], "weight") and pretrain:
                self.ncfpp.mlp.mlp[i].weight.requires_grad = False
            if hasattr(self.ncfpp.mlp.mlp[i], "bias") and pretrain:
                self.ncfpp.mlp.mlp[i].bias.requires_grad = False
            if type(module) == torch.nn.Linear:
                if module.out_features == 1:
                    # Add NCFPP last hidden input dim to `input_dim`
                    input_dim += module.in_features 
                    break
            ncfpp_mlp_blocks.append(module)
        self.ncfpp.mlp = torch.nn.Sequential(*ncfpp_mlp_blocks)
        
        # Singlelayer perceptrons
        # input_dim = GMF embedding dim + ncfpp last hidden input dim
        self.slp = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")

    def forward(self, x):
        # GMF part
        gmf_x_embed = self.gmf.embedding(x)
        gmf_user_embed = gmf_x_embed[:, 0]
        gmf_item_embed = gmf_x_embed[:, 1]
        # Element wise product between embeddings
        gmf_product = gmf_user_embed *  gmf_item_embed

        # NCF part
        # Get output of last hidden layer
        ncf_last_hidden = self.ncf(x)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.slp(concat)

        return y

class NMFPPModule(BaseModule):
    def __init__(self, gmfpp:GMFPP, ncfpp:NCFPP,
                 pretrain:bool = False,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NMFPP(gmfpp, ncfpp, pretrain)
        self.save_hyperparameters()

    def evaluate(self, x, y): 
        # Partition user and item features
        user_feature = x[:, 2:self.model.user_feature_dim + 2]
        item_feature = x[:, -self.model.item_feature_dim:]

        loss = 0
        loss += self.criterion(self.model.gmfpp.user_ae.forward(user_feature), user_feature)
        loss += self.criterion(self.model.gmfpp.item_ae.forward(item_feature), item_feature)
        loss += self.criterion(self.model.ncfpp.user_ae.forward(user_feature), user_feature)
        loss += self.criterion(self.model.ncfpp.item_ae.forward(item_feature), item_feature)
        loss += super().evaluate(x, y)

        return loss