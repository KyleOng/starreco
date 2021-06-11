from os import remove
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .module import BaseModule
from .layers import MultilayerPerceptrons, ActivationFunction, FeaturesEmbedding

# Done
class CMF(BaseModule):
    """
    Convolutional Matrix Factorization.

    - user_dim (int): User dimension.
    - word_dim (int): Word/vocabulary dimension.
    - max_len (int): Max sentence length.
    - user_embed_dim (int): User embedding dimension.
    - word_embed_dim (int): Word/vocabulary embedding dimension.
    - filter_size (int): Convolution filter/depth/channel size. Default: 100.
    - kernel_size (int): Convolution square-window size or convolving square-kernel size. Default: 3
    - c_activation (str): Activation function applied across the convolution layers. Default: "relu".
    - c_dropout (int/float): Convolution dropout rate. Default: 0.
    - fc_dim (int): Fully connected layer dimension. Default: 200.
    - fc_activation (str): Fully connected layer activation function. Default: "tanh".
    - fc_dropout (int/float): Fully connected layer dropout rate. Default: 0.
    - batch_norm (bool): If True, apply batch normalization during convolutions and fully connected layer. Batch normalization is applied between activation and dropout layer across the convolution layers. Default: True.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.

    Note: CNN is used to address the problem of varying sentences length by taking the maximum (pooling) of the convoluted sentence embeddings.
    """

    def __init__(self, 
                 user_dim:int,
                 word_dim:int,
                 max_len:int, 
                 user_embed_dim:int = 50,
                 word_embed_dim:int = 200,
                 filter_size:list = 100,
                 kernel_size:int = 3,
                 c_activation:str = "relu",
                 c_dropout:Union[int, float] = 0,
                 fc_dim:int = 200,
                 fc_activation:str = "tanh", 
                 fc_dropout:Union[int, float] = 0.2,
                 batch_norm:bool = True,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-6,
                 criterion = F.mse_loss):
        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        # Features embedding layer
        # User embedding layer
        self.user_embedding = torch.nn.Embedding(user_dim, user_embed_dim)
        # Word embedding layer
        self.word_embedding = torch.nn.Embedding(word_dim, word_embed_dim)

        # Convolution layer
        cnn_blocks = []
        # Convolution 
        convolution = torch.nn.Conv2d(1, filter_size, (kernel_size, word_embed_dim))
        cnn_blocks.append(convolution)

        # Batch normalization
        if batch_norm:
            batch_normalization = torch.nn.BatchNorm2d(filter_size)
            cnn_blocks.append(batch_normalization)

        # Activation function
        activation_fucnction = ActivationFunction(c_activation)
        cnn_blocks.append(activation_fucnction)

        # Pooling layer
        pooling = torch.nn.MaxPool2d((max_len - kernel_size + 1, 1))
        cnn_blocks.append(pooling)

        # Dropout
        if c_dropout > 0 and c_dropout < 1:
            cnn_blocks.append(torch.nn.Dropout(c_dropout))

        # Flatten
        cnn_blocks.append(torch.nn.Flatten())   

        # Fully connected layer
        net = MultilayerPerceptrons(input_dim = filter_size, 
                                            hidden_dims = [fc_dim, user_embed_dim], 
                                            activations = fc_activation, 
                                            dropouts = fc_dropout,
                                            remove_last_batch_norm = True,
                                            remove_last_dropout = True,
                                            output_layer = None,
                                            batch_norm = batch_norm)
        cnn_blocks.append(net)

        self.cnn = torch.nn.Sequential(*cnn_blocks)
        
        
    def load_pretrain_embeddings(self, vocab_map, glove_path = "glove.6B/glove.6B.200d.txt"):
        """
        Load pretrain word embeddings.
        """
        # Load pretrained word embeddings model.
        word_embeddings = {}
        num_lines = sum(1 for _ in open(glove_path, 'r', encoding="utf-8"))
        with open(glove_path, 'r', encoding="utf-8") as f:
            f_tqdm = tqdm(f, total = num_lines)
            f_tqdm.set_description("loading pretrained")
            for line in f_tqdm:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                word_embeddings[word] = vector

        # Load pretrained word embeddings to embedding layer
        # Get unpretrained embeddings
        unpretrained_embeddings = self.word_embedding.weight.detach().numpy()
        # Pretrained embeddings starts from 2, as 1s are non-vocabs for and 0s for padding.
        pretrained_embeddings = [unpretrained_embeddings[0], unpretrained_embeddings[1]]
        embed_dim = self.word_embedding.embedding_dim
        vocab_map_tqdm = tqdm(enumerate(vocab_map.keys(), start = 2))
        vocab_map_tqdm.set_description("prepare_embeddings")
        for i, vocab in vocab_map_tqdm:
            pretrained_embedding = word_embeddings[vocab][:embed_dim]
            pretrained_embeddings.append(pretrained_embedding)

        # Load pretrained word embeddings to embedding layer
        self.word_embedding = self.word_embedding.from_pretrained(torch.tensor(pretrained_embeddings))

    def forward(self, x, _, word):
        # Get user embeddings
        user_x = x[:,0].int()
        user_embed = self.user_embedding(user_x)

        # Get context aware item embeddings
        word_embed = self.word_embedding(word.int())
        word_embed = word_embed.unsqueeze(1)
        item_embed = self.cnn(word_embed)

        # Dot product between user and item embeddings
        dot = torch.sum(user_embed * item_embed, dim = 1)
        
        # Reshape to match target shape
        y = dot.view(dot.shape[0], -1)

        return y
