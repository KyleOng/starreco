from typing import Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .module import BaseModule
from .layer import FeaturesEmbedding
from .mf import MF

class MDACF(MF):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering
    """

    def __init__(self, 
                 user_dims:list, 
                 item_dims:list,
                 embed_dim:int,
                 corrupt_ratio:Union[int,float],
                 lambda_:Union[int,float]):

        # Using the same notation as in the paper for easy reference
        self.m, self.p = user_dims
        self.n, self.q = item_dims 
        self.corrupt_ratio = corrupt_ratio
        self.lambda_ = lambda_

        super().__init__([self.m, self.n], embed_dim)

        # marginalized Autoencoder
        # Create weights matrices and projection matrices
        self.user_W = torch.rand(self.p, self.p)
        self.user_P = torch.rand(self.p, embed_dim)
        self.item_W = torch.rand(self.q, self.q)
        self.item_P = torch.rand(self.q, embed_dim)

        # Get latent factors
        # Obtain values only without gradient
        self.U = self.embedding.embedding.weight[:self.m, :].data
        self.V = self.embedding.embedding.weight[-self.n:, :].data
            
    def _update_projections(self, 
                            X:torch.Tensor, 
                            W:torch.Tensor, 
                            U:torch.Tensor):
        """
        Projection matrix optimization.
        """

        # P=A/B
        # A=U'U
        A = torch.matmul(U.T, U)
        # B=WXU
        B = torch.matmul(torch.matmul(W, X), U)
        return torch.linalg.solve(A.T, B.T).T

    def _update_weights(self, 
                        X:torch.Tensor, 
                        P:torch.Tensor, 
                        U:torch.Tensor, 
                        d:int, 
                        lambda_:Union[int, float] = 0.3, 
                        p:Union[int, float] = 0.3):
        """
        Weight matrix optimization. 
        Note: The weight optimization algorithm is inspired by mDA.
        """

        # mDA pseudo code in MATLAB
        # MATLAB: q=[ones(d-1,1).*(1-p); 1];
        q = torch.ones((d, 1)) * (1 - p)
        # MATLAB: S=X*X’;
        S = torch.matmul(X, X.T)
        # MATLAB: Q=S.*(q*q’);
        Q = S * torch.matmul(q, q.T)
        # MATLAB: Q(1:d+1:end)=q.*diag((X*X’));
        v = q[:,0] * S.diag()
        mask = torch.diag(torch.ones_like(v))
        Q = mask * torch.diag(v) + (1 - mask) * Q
        # MATLAB: (Q+1e-5*eye(d))
        Q += 1e-5 *torch.eye(d)
        # MATLAB: S=S.*repmat(q’,d,1);
        S *= torch.tile(q.T, (d, 1)) # torch.tile requires torch version 1.8 and above

        # W=E[S]/E[Q]
        # E[S]=S+λU'X'
        ES = S.detach().clone()
        ES += lambda_ * torch.matmul(P, torch.matmul(U.T, X.T))
        # E[Q]=S+λQ
        EQ = S.detach().clone()
        EQ += lambda_ * Q
        return torch.linalg.solve(EQ, ES)

    def forward(self, 
                x:torch.Tensor, 
                user:torch.Tensor, 
                item:torch.Tensor):
        
        # Marginalized Autoencoder
        # Update weights and projection matrices only in training mode
        if self.training:
            # Defined X (user feature) and Y (item feature), notation same as in paper
            if user.shape != [self.m, self.p]:
                X = user.T  
            else:
                X = user

            if item.shape != [self.n, self.q]:
                Y = item.T 
            else:
                Y = item

            # Update weights and projections matrices
            self.user_W = self._update_weights(X, self.user_P, self.U, self.p, self.lambda_, self.corrupt_ratio)
            self.item_W = self.update_weights(Y, self.item_P, self.V, self.q, self.lambda_, self.corrupt_ratio)
            self.user_P = self.update_projections(X, self.user_W, self.U)
            self.item_P = self.update_projections(Y, self.item_W, self.V)

            # Get new latent factors
            # Obtain values only without gradient
            self.U = self.embedding.embedding.weight[:self.m, :].data.to("cpu")
            self.V = self.embedding.embedding.weight[-self.n:, :].data.to("cpu")

        y = super().forward(x)

        return y

class MDACFModule(BaseModule):
    def __init__(self, 
                 user:torch.Tensor, 
                 item:torch.Tensor,
                 embed_dim:int = 8, 
                 corrupt_ratio:Union[int,float] = 0.3,
                 lambda_:Union[int,float] = 0.3,
                 alpha:Union[int,float] = 0.8, 
                 beta:Union[int,float] = 3e-3,
                 mean:bool = False,
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):
        super().__init__(lr, weight_decay, criterion)
        self.model = MDACF(user, item, embed_dim, corrupt_ratio, lambda_)
        self.alpha = alpha
        self.beta = beta
        self.mean = mean
        self.save_hyperparameters()

    def evaluate(self, 
                 x:torch.Tensor,
                 user:torch.Tensor, 
                 item:torch.Tensor 
                 y:torch.Tensor):
        torch_fn = torch.mean if self.mean else torch.sum
        
        loss = 0
        loss += self.model.lambda_ * torch_fn(torch.square(torch.matmul(self.model.user_P, self.model.U.T) - torch.matmul(self.model.user_W, user)))
        loss += self.model.lambda_ * torch_fn(torch.square(torch.matmul(self.model.item_P, self.model.V.T) - torch.matmul(self.model.item_W, item)))
        loss += self.alpha * torch_fn(torch.square((y - super(type(self.model), self.model).forward(x)).to("cpu")))
        loss += self.beta * (torch_fn(torch.square(self.model.U)) + torch_fn(torch.square(self.model.V)))
        loss = loss.to(self.device)
        return loss