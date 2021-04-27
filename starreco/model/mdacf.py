from typing import Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .mf import MF

class _MDACF(MF):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering
    """

    def __init__(self, 
                 user_dims:list, 
                 item_dims:list,
                 embed_dim:int, 
                 corrupt_ratio:Union[int,float],
                 alpha:Union[int,float], 
                 beta:Union[int,float],
                 lambda_:Union[int,float],
                 mean:bool,
                 lr:float,
                 weight_decay:float,
                 criterion:F):

        # Using the same notation as in the paper for easy reference
        self.m, self.p = user_dims
        self.n, self.q = item_dims 
        self.corrupt_ratio = corrupt_ratio
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.mean = mean

        super().__init__([self.m, self.n], embed_dim, lr, weight_decay, criterion)

        # Create weights matrices and projection matrices for marginalized Autoencoder.
        self.user_W = torch.rand(self.p, self.p)
        self.user_P = torch.rand(self.p, embed_dim)
        self.item_W = torch.rand(self.q, self.q)
        self.item_P = torch.rand(self.q, embed_dim)

        # Get latent factors
        self.U = self.features_embedding.embedding.weight[:self.m, :].data
        self.V = self.features_embedding.embedding.weight[-self.n:, :].data

        self.save_hyperparameters()
            
    def _update_projections(_, 
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

    def _update_weights(_, 
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
                x:torch.Tensor):
        
        # Marginalized Autoencoder
        # Update weights and projection matrices only in training mode
        if self.training:
            # Update weights and projections matrices
            self.user_W = self._update_weights(self.user.T, self.user_P, self.U, self.p, self.lambda_, self.corrupt_ratio)
            self.item_W = self._update_weights(self.item.T, self.item_P, self.V, self.q, self.lambda_, self.corrupt_ratio)
            self.user_P = self._update_projections(self.user.T, self.user_W, self.U)
            self.item_P = self._update_projections(self.item.T, self.item_W, self.V)

            # Get latent factors
            self.U = self.features_embedding.embedding.weight[:self.m, :].data
            self.V = self.features_embedding.embedding.weight[-self.n:, :].data

        y = super().forward(x)
        return y

    def backward_loss(self, *batch):
        x, y = batch
        torch_fn = torch.mean if self.mean else torch.sum
            
        loss = 0
        loss += self.lambda_ * torch_fn(torch.square(torch.matmul(self.user_P, self.U.T) - torch.matmul(self.user_W, self.user.T)))
        loss += self.lambda_ * torch_fn(torch.square(torch.matmul(self.item_P, self.V.T) - torch.matmul(self.item_W, self.item.T)))
        loss += self.alpha * torch_fn(torch.square(y - super().forward(x)))
        loss += self.beta * (torch_fn(torch.square(self.U)) + torch_fn(torch.square(self.V)))

        return loss

    def logger_loss(self, *batch):
        y_hat = super().forward(*batch[:-1])
        loss = self.criterion(y_hat, batch[-1])

        return loss

        
class MDACF(_MDACF):
    def __init__(self, 
                 user:torch.Tensor, 
                 item:torch.Tensor,
                 embed_dim:int = 8, 
                 corrupt_ratio:Union[int,float] = 0.3,
                 alpha:Union[int,float] = 0.8, 
                 beta:Union[int,float] = 3e-3,
                 lambda_:Union[int,float] = 0.3,
                 mean:bool = False,
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):
        super().__init__(user.shape, item.shape, embed_dim, corrupt_ratio, alpha, beta, lambda_, mean, lr, weight_decay, criterion)
        # Need to change
        self.user = user
        self.item = item
