from typing import Union

import torch
import torch.nn.functional as F

from .mf import MF

# Done
class _MDACF(MF):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering.

    - user_dims (list): User matrix dimension.
    - item_dins (list): Item matrix dimension.
    - embed_dim (int): Embedding dimension.
    - corrupt_ratio (int/float): Probability of noises to be added to the input.
    - alpha (int/float): Trade-off parameter value for user SDAE.
    - beta (int/float): Trade-off parameter value for item SDAE.
    - lr (float): Learning rate.
    - l2_lambda (float): L2 regularization rate.
    - criterion (F): Criterion or objective or loss function.

    Warning: This method should not be used directly.
    """

    def __init__(self, 
                 user_dims:list, 
                 item_dims:list,
                 embed_dim:int, 
                 corrupt_ratio:Union[int,float],
                 alpha:Union[int,float], 
                 beta:Union[int,float],
                 lr:float,
                 l2_lambda:Union[int,float],
                 criterion:F):
        self.m, self.p = user_dims
        self.n, self.q = item_dims 

        super().__init__([self.m, self.n], embed_dim, lr, 0, criterion)
        self.save_hyperparameters()

        self.corrupt_ratio = corrupt_ratio
        self.alpha = alpha
        self.beta = beta
        self.l2_lambda = l2_lambda

        # Create weights matrices and projection matrices for marginalized Autoencoder.
        self.user_W = torch.nn.Parameter(torch.rand(self.p, self.p), requires_grad = False)
        self.user_P = torch.nn.Parameter(torch.rand(self.p, embed_dim), requires_grad = False)
        self.item_W = torch.nn.Parameter(torch.rand(self.q, self.q), requires_grad = False)
        self.item_P = torch.nn.Parameter(torch.rand(self.q, embed_dim), requires_grad = False)

    def _update_projections(_, X, W, U):
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

    def _update_weights(_, X, P, U, d, lambda_, p):
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
        self.user_W.data = self._update_weights(self.user.T, user_P, U, self.p, self.l2_lambda, self.corrupt_ratio)
        self.item_W.data = self._update_weights(self.item.T, item_P, V, self.q, self.l2_lambda, self.corrupt_ratio)
        self.user_P.data = self._update_projections(self.user.T, user_W, U)
        self.item_P.data = self._update_projections(self.item.T, item_W, V)

    def backward_loss(self, *batch):
        x, y = batch

        # Transfer data to `self.device` during the 1st epoch (validation)
        if self.current_epoch == 0:
            self.user = self.user.to(self.device)
            self.item = self.item.to(self.device)

        # Update W and P during training via MDAE
        if self.training:
            self.mdae()

        user_W = self.user_W.data
        item_W = self.item_W.data
        user_P = self.user_P.data
        item_P = self.item_P.data
        U = self.features_embedding.embedding.weight[:self.m, :].data
        V = self.features_embedding.embedding.weight[-self.n:, :].data
        
        # User reconstruction loss
        user_loss = self.l2_lambda * torch.sum(torch.square(torch.matmul(user_P, U.T) - torch.matmul(user_W, self.user.T)))
        # Item reconstruction loss
        item_loss = self.l2_lambda * torch.sum(torch.square(torch.matmul(item_P, V.T) - torch.matmul(item_W, self.item.T)))
        # Rating loss
        rating_loss =  self.alpha * torch.sum(torch.square(y - self.forward(x)))

        loss = rating_loss + user_loss + item_loss 
        loss += self.beta * (torch.sum(torch.square(U)) + torch.sum(torch.square(V)))

        return loss

# Future work: allow sparse matrix factorization.
class MDACF(_MDACF):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering (private).

    - user (torch.Tensor): User matrix.
    - item (torch.Tensor): Item matrix.
    - embed_dim (int): Embedding dimension. Default: 8.
    - corrupt_ratio (int/float): Probability of noises to be added to the input. Default: 0.3.
    - alpha (int/float): Trade-off parameter value for user SDAE. Default: 0.8.
    - beta (int/float): Trade-off parameter value for item SDAE. Default: 3e-3.
    - lr (float): Learning rate. Default: 1e-3.
    - l2_lambda (float): L2 regularization rate. Default: 1e-3.
    - criterion (F): Criterion or objective or loss function. Default: F.mse_loss.  
    """

    def __init__(self, 
                 user:torch.Tensor, 
                 item:torch.Tensor,
                 embed_dim:int = 8, 
                 corrupt_ratio:Union[int,float] = 0.3,
                 alpha:Union[int,float] = 0.8, 
                 beta:Union[int,float] = 3e-3,
                 lr:float = 1e-3,
                 l2_lambda:Union[int,float] = 0.3,
                 criterion:F = F.mse_loss):
        super().__init__(user.shape, item.shape, embed_dim, corrupt_ratio, alpha, beta, lr, l2_lambda, criterion)
        self.user = user
        self.item = item