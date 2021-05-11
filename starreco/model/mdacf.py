from typing import Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .mf import MF

  
class _MDACF(MF):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering model
    """

    def __init__(self, 
                 user_dims:list, 
                 item_dims:list,
                 embed_dim:int = 8, 
                 corrupt_ratio:Union[int,float] = 0.3,
                 alpha:Union[int,float] = 0.8, 
                 beta:Union[int,float] = 3e-3,
                 lambda_:Union[int,float] = 0.3,
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):
        # Using the same notation as in the paper for easy reference
        self.m, self.p = user_dims
        self.n, self.q = item_dims 
        self.corrupt_ratio = corrupt_ratio
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

        super().__init__([self.m, self.n], embed_dim, lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Create weights matrices and projection matrices for marginalized Autoencoder.
        self.user_W = torch.nn.Parameter(torch.rand(self.p, self.p), requires_grad = False)
        self.user_P = torch.nn.Parameter(torch.rand(self.p, embed_dim), requires_grad = False)
        self.item_W = torch.nn.Parameter(torch.rand(self.q, self.q), requires_grad = False)
        self.item_P = torch.nn.Parameter(torch.rand(self.q, embed_dim), requires_grad = False)

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
        P = torch.linalg.solve(A.T, B.T).T

        # Clamp between -1 and 1
        return torch.clamp(P, min = -1, max = 1)

    def _update_weights(_, 
                        X:torch.Tensor, 
                        P:torch.Tensor, 
                        U:torch.Tensor, 
                        d:int, 
                        lambda_:Union[int, float],
                        p:Union[int, float]):
        """
        Weight matrix optimization. 
        Note: The weight optimization algorithm is inspired by mDA.
        """
        device = X.device and P.device and U.device

        # mDA pseudo code in MATLAB
        # MATLAB: q=[ones(d-1,1).*(1-p); 1];
        q = torch.ones((d, 1)).to(device) * (1 - p)
        # MATLAB: S=X*X’;
        S = torch.matmul(X, X.T)
        # MATLAB: Q=S.*(q*q’);
        Q = S * torch.matmul(q, q.T)
        # MATLAB: Q(1:d+1:end)=q.*diag((X*X’));
        v = q[:,0] * S.diag()
        mask = torch.diag(torch.ones_like(v))
        Q = mask * torch.diag(v) + (1 - mask) * Q
        # MATLAB: (Q+1e-5*eye(d))
        Q += 1e-5 *torch.eye(d).to(device)
        # MATLAB: S=S.*repmat(q’,d,1);
        S *= torch.tile(q.T, (d, 1)) # torch.tile requires torch version 1.8 and above

        # W=E[S]/E[Q]
        # E[S]=S+λU'X'
       
        ES = S.detach().clone().to(device)
        ES += lambda_ * torch.matmul(P, torch.matmul(U.T, X.T))
        # E[Q]=S+λQ
        EQ = S.detach().clone().to(device)
        EQ += lambda_ * Q
        W = torch.linalg.solve(EQ, ES)

        # Clamp between -1 and 1
        return torch.clamp(W, min = -1, max = 1)

    def mdae(self):
        # Marginalized Autoencoder
        # Update weights and projection matrices only in training mode

        # Get data from Parameters
        user_W = self.user_W.data
        item_W = self.item_W.data
        user_P = self.user_P.data
        item_P = self.item_P.data
        U = self.features_embedding.embedding.weight[:self.m, :].data
        V = self.features_embedding.embedding.weight[-self.n:, :].data

        # Update weights and projections matrices
        self.user_W.data = self._update_weights(self.user.T, user_P, U, self.p, self.lambda_, self.corrupt_ratio)
        self.item_W.data = self._update_weights(self.item.T, item_P, V, self.q, self.lambda_, self.corrupt_ratio)
        self.user_P.data = self._update_projections(self.user.T, user_W, U)
        self.item_P.data = self._update_projections(self.item.T, item_W, V)

    def backward_loss(self, *batch):
        x, y = batch

        if self.current_epoch == 0:
            self.user = self.user.to(self.device)
            self.item = self.item.to(self.device)

        if self.training:
            self.mdae()

        user_W = self.user_W.data
        item_W = self.item_W.data
        user_P = self.user_P.data
        item_P = self.item_P.data
        U = self.features_embedding.embedding.weight[:self.m, :].data
        V = self.features_embedding.embedding.weight[-self.n:, :].data
        
        loss = 0
        loss += self.lambda_ * torch.sum(torch.square(torch.matmul(user_P, U.T) - torch.matmul(user_W, self.user.T)))
        loss += self.lambda_ * torch.sum(torch.square(torch.matmul(item_P, V.T) - torch.matmul(item_W, self.item.T)))
        loss += self.alpha * torch.sum(torch.square(y - self.forward(x)))
        loss += self.beta * (torch.sum(torch.square(U)) + torch.sum(torch.square(V)))

        return loss

    def logger_loss(self, *batch):
        x, y = batch

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        return loss


class MDACF(_MDACF):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering lightning module
    """

    def __init__(self, 
                 user:torch.Tensor, 
                 item:torch.Tensor,
                 embed_dim:int = 8, 
                 corrupt_ratio:Union[int,float] = 0.3,
                 alpha:Union[int,float] = 0.8, 
                 beta:Union[int,float] = 3e-3,
                 lambda_:Union[int,float] = 0.3,
                 lr:float = 1e-3,
                 weight_decay:float = 0,
                 criterion:F = F.mse_loss):
        super().__init__(user.shape, item.shape, embed_dim, corrupt_ratio, alpha, beta, lambda_, lr, weight_decay, criterion)
        self.user = user
        self.item = item