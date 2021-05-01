from typing import Union
import copy

import torch
import torch.nn.functional as F

from .module import BaseModule
from .layer import FeaturesEmbedding, MultilayerPerceptrons
from .gmfpp import GMFPPmodel
from .ncfpp import NCFPPmodel

class NMFPPmodel(torch.nn.Module):
    """
    Neural Matrix Factorizatgion ++ model
    """
    def __init__(self, 
                 gmfpp:GMFPPmodel, 
                 ncfpp:NCFPPmodel,
                 freeze_pretrain:bool):
        super().__init__()

        # Clone pretrained models
        self.gmfpp = copy.deepcopy(gmfpp)
        self.ncfpp = copy.deepcopy(ncfpp)

        # Freeze model
        if freeze_pretrain:
            for param in self.gmfpp.parameters():
                param.requires_grad = False
            for param in self.ncfpp.parameters():
                param.requires_grad = False

        # Remove GMF output layer
        del self.gmfpp.net

        # Remove NCF output layer
        del self.ncfpp.net.mlp[-1]
        if type(self.ncfpp.net.mlp[-1]) == torch.nn.Linear:
            del self.ncfpp.net.mlp[-1]

        # Get input dim 
        input_dim = self.gmfpp.features_embedding.embedding.embedding_dim 
        input_dim += self.gmfpp.user_ae.decoder.mlp[0].in_features and self.gmfpp.item_ae.decoder.mlp[0].in_features 
        for i in range(0, len(self.ncfpp.net.mlp))[::-1]:
            if type(self.ncfpp.net.mlp[i]) == torch.nn.Linear:
                input_dim += self.ncfpp.net.mlp[i].out_features
                break
        # Hybrid network
        self.net = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")

    def forward(self, x, user_x, item_x):
        # GMF part: Element wise product between embeddings
        gmf_product = self.gmfpp.encode_element_wise_product(x, user_x, item_x)

        # NCF part: Get output of last hidden layer
        ncf_last_hidden = self.ncfpp.forward(x, user_x, item_x)

        # Concatenate GMF's element wise product and NCF's last hidden layer output
        concat = torch.cat([gmf_product, ncf_last_hidden], dim = 1)

        # Feed to generalized non-linear layer
        y = self.net(concat)

        return y


class NMFPP(BaseModule):
    def __init__(self, 
                 gmfpp:GMFPPmodel, 
                 ncfpp:NCFPPmodel,
                 freeze_pretrain:bool = True,
                 alpha:Union[int,float] = 0.1, 
                 beta:Union[int,float] = 0.1,
                 gamma:Union[int,float] = 0.1, 
                 delta:Union[int,float] = 0.1,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = NMFPPmodel(gmfpp, ncfpp, freeze_pretrain)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.save_hyperparameters()
    
    def forward(self, *batch):
        return self.model.forward(*batch)

    def backward_loss(self, *batch):
        x, user_x, item_x, y = batch

        loss = 0
        loss += self.alpha * self.criterion(self.model.gmfpp.user_ae.forward(user_x), user_x)
        loss += self.beta * self.criterion(self.model.gmfpp.item_ae.forward(item_x), item_x)
        loss += self.gamma * self.criterion(self.model.ncfpp.user_ae.forward(user_x), user_x)
        loss += self.delta * self.criterion(self.model.ncfpp.item_ae.forward(item_x), item_x)
        loss += super().backward_loss(*batch)

        return loss