from typing import Union

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from starreco.model import (Module,
                            FeaturesEmbedding)

class MDACF(Module):
    """
    Marginalized Denoising Autoencoder Collaborative Filtering
    """
    def __init__(self, user_feature:torch.Tensor, item_feature:torch.Tensor,
                 embed_dim:int = 8, 
                 alpha:Union[int,float] = 0.8, 
                 beta:Union[int,float] = 3e-3,
                 lambda_:Union[int,float] = 0.3,
                 corrupt_ratio:Union[int,float] = 0.3,
                 mean:bool = False,
                 lr:float = 1e-2,
                 weight_decay:float = 1e-6,
                 criterion:F = F.mse_loss,
                 save_hyperparameters:bool = True):

        super().__init__(lr, weight_decay, criterion)

        # Using notation same as paper for easy reference
        self.m, self.p = user_feature.shape
        self.n, self.q = item_feature.shape
        self.X = user_feature.T
        self.Y = item_feature.T
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.corrupt_ratio = corrupt_ratio
        self.mean = mean

        # marginalized Autoencoder
        # Create weights matrices and projection matrices
        self.user_W = torch.rand(self.p, self.p)
        self.user_P = torch.rand(self.p, embed_dim)
        self.item_W = torch.rand(self.q, self.q)
        self.item_P = torch.rand(self.q, embed_dim)
        
        # Embedding layer
        self.embedding = FeaturesEmbedding([self.m, self.n], embed_dim)
        self.U = self.embedding.embedding.weight[:self.m, :].detach().clone()
        self.V = self.embedding.embedding.weight[-self.n:, :].detach().clone()

        # Save hyperparameters to checkpoint
        if save_hyperparameters:
            self.save_hyperparameters()

    def update_projections(_, X, W, U):
        """
        Projection matrix optimization.
        """

        # P=A/B
        # A=U'U
        A = torch.matmul(U.T, U)
        # B=WXU
        B = torch.matmul(torch.matmul(W, X), U)
        return torch.linalg.solve(A.T, B.T).T

    def update_weights(_, X, P, U, d, lambda_ = 0.3, p = 0.3):
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

    def forward(self, x):
        """
        Perform operations.

        :x (torch.tensor): Input tensors of shape (batch_size, 2) user and item.

        :return (torch.tensor): Output prediction tensors of shape (batch_size, 1)
        """
        # Marginalized Autoencoder
        # Update weights and projection matrices only in training mode
        if self.training:
            self.user_W = self.update_weights(self.X, self.user_P, self.U, self.p, self.lambda_, self.corrupt_ratio)
            self.item_W = self.update_weights(self.Y, self.item_P, self.V, self.q, self.lambda_, self.corrupt_ratio)
            self.user_P = self.update_projections(self.X, self.user_W, self.U)
            self.item_P = self.update_projections(self.Y, self.item_W, self.V)
        
        # Generate embeddings
        x = self.embedding(x)
        # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Dot product between user and items embeddings
        dot = torch.mm(user_embedding, item_embedding.T)
        # Obtain diagonal part of the dot product result
        diagonal = dot.diag()
        # Reshape to match evaluation shape
        y = diagonal.view(diagonal.shape[0], -1)

        return y

    def evaluate(self, x, y):
        torch_fn = torch.mean if self.mean else torch.sum
        loss = 0
        loss += self.lambda_ * torch_fn(torch.square(torch.matmul(self.user_P, self.U.T) - torch.matmul(self.user_W, self.X)))
        loss += self.lambda_ * torch_fn(torch.square(torch.matmul(self.item_P, self.V.T) - torch.matmul(self.item_W, self.Y)))
        loss += self.alpha * torch_fn(torch.square((y - self.forward(x)).to("cpu")))
        loss += self.beta * (torch_fn(torch.square(self.U)) + torch_fn(torch.square(self.V)))
        loss = loss.to(self.device)
        return torch_fn(torch.square(self.forward(x) - y))
        #loss = F.mse_loss(self.forward(x), y)
        return loss