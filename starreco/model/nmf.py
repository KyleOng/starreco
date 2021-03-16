import torch
import torch.nn.functional as F

from starreco.model import FeaturesEmbedding, MultilayerPerceptrons, Module

class NMF(Module):
    def __init__(self, features_dim, embed_dim,
                 ncf_output_layers, ncf_activations, ncf_dropouts,
                 nmf_output_layers, nmf_activations, nmf_dropouts,
                 ncf_batch_norm = True, nmf_batch_norm = True, 
                 criterion = F.mse_loss):
        super().__init__()
        self.criterion = criterion

        # Embedding layer
        self.embedding = FeaturesEmbedding(features_dim, embed_dim)

        # NCF MLP layers
        ncf_output_layers.insert(0, embed_dim * 2)
        self.ncf_mlp = MultilayerPerceptrons(ncf_output_layers, 
                                             ncf_activations, 
                                             ncf_dropouts,
                                             ncf_batch_norm)

        # NMF NML layers
        # Number of NMF's input nodes = number of NCF's output nodes + 1
        nmf_output_layers.insert(0, ncf_output_layers[-1] + 1)
        self.nmf_mlp = MultilayerPerceptrons(nmf_output_layers, 
                                             nmf_activations, 
                                             nmf_dropouts,
                                             nmf_batch_norm)

    def forward(self, x):
        # Generate embeddings
        x = self.embedding(x)
        user_embedding = x[:, 0]
        item_embedding = x[:, 1]

        # Matrix Factorization
        # Dot product between embedding
        matrix = torch.mm(user_embedding, item_embedding.T)
        # Obtain diagonal part of the outer product result
        diagonal = matrix.diag()
        # Reshape
        mf = diagonal.view(diagonal.shape[0], -1)

        # Neural Collaborative Filtering
        # Concatenate embeddings 
        concat = torch.flatten(x, start_dim = 1)
        # NCF prediction
        ncf = self.ncf_mlp(concat)

        # Neural Matrix Factorization
        # Concatenate result from MF and NCF
        combine = torch.cat((mf, ncf), 1)
        # NMF prediction
        return self.nmf_mlp(combine)