import torch
import torch.nn.functional as F

from .module import BaseModule
from .gmf import GMF
from .ncf import NCF
from .layers import MultilayerPerceptrons

# Done
class NMF(BaseModule):
    """
    Neural Matrix Factorization.

    - gmf_kwargs (dict): GMF keyword arguments (hyperparameters).
    - ncf_kwargs (dict): NCF keyword arguments (hyperparameters).
    - shared_embed (str): Model which to share feature embeddings. Default: None.
        - If "gmf", GMF and NCF embeddings will be obtained from GMF features embedding layer, NCF features embedding layer will be deleted/ignored.
        - If "ncf", GMF and NCF embeddings will be obtained from NCF features embedding layer, GMF features embedding layer will be deleted/ignored.
        - If None, GMF and NCF will perform feature embeddings seperately.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """
    
    def __init__(self, 
                 gmf_kwargs:dict,
                 ncf_kwargs:dict,
                 shared_embed:str = None,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        assert shared_embed in [None, "gmf", "ncf"], "`shared_embed` must be either None, 'gmf' or 'ncf'."

        super().__init__(lr, weight_decay, criterion)
        self.save_hyperparameters()

        self.shared_embed = shared_embed
        
        self.gmf = GMF(**gmf_kwargs)
        self.ncf = NCF(**ncf_kwargs)

        # If `shared_embed` is not None, delete one of the feature embedding layer based on arg value.
        if shared_embed == "gmf":
            del self.ncf.features_embedding
        elif shared_embed == "ncf":     
            del self.gmf.features_embedding    

        # Remove GMF output layer
        del self.gmf.net

        # Remove NCF output layer
        del self.ncf.net.mlp[-1]
        if type(self.ncf.net.mlp[-1]) == torch.nn.Linear:
            del self.ncf.net.mlp[-1]

        # Multilayer Perceptrons layer
        input_dim = gmf_kwargs["embed_dim"] + ncf_kwargs["hidden_dims"][-1]
        self.net = MultilayerPerceptrons(input_dim = input_dim, 
                                         output_layer = "relu")

    def fusion(self, x):
        if self.shared_embed == "gmf":
            # Share embeddings from GMF features embedding layer.
            x_embed_gmf = x_embed_ncf = self.gmf.features_embedding(x.int())
        elif self.shared_embed == "ncf":
            # Share embeddings from NCF features embedding layer.
            x_embed_gmf = x_embed_ncf = self.ncf.features_embedding(x.int())
        else: # None
            # Seperate feature embeddings between GMF and NCF
            x_embed_gmf = self.gmf.features_embedding(x.int())
            x_embed_ncf = self.ncf.features_embedding(x.int())
        
        # GMF interaction function
        # Element wise product between user and item embeddings
        user_embed, item_embed = x_embed_gmf[:, 0], x_embed_gmf[:, 1]
        embed_product = user_embed * item_embed
        output_gmf = embed_product

        # NCF interaction function
        # Concatenate user and item embeddings
        embed_concat = torch.flatten(x_embed_ncf, start_dim = 1)
        # Non linear on concatenated user and item embeddings
        output_ncf = self.ncf.net(embed_concat)

        # Concatenate GMF element wise product and NCF last hidden layer output
        fusion = torch.cat([output_gmf, output_ncf], dim = 1)

        return fusion

    def forward(self, x):
        # Fusion of GMF and NCF
        fusion = self.fusion(x)

        # Non linear on fusion vectors
        y = self.net(fusion)

        return y

    def load_pretrain_weights(self, gmf_weights, ncf_weights, freeze = True):
        """
        Load pretrain weights for GMF and NCF.
        """
        gmf_weights = {k:v for k,v in gmf_weights.items() if k in self.gmf.state_dict()}
        ncf_weights = {k:v for k,v in ncf_weights.items() if k in self.ncf.state_dict()}

        self.gmf.load_state_dict(gmf_weights)
        self.ncf.load_state_dict(ncf_weights)

        if freeze:
            self.gmf.freeze()
            self.ncf.freeze()     