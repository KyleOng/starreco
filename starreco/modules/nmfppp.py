import torch
import torch.nn.functional as F

from .module import BaseModule
from .nmf import NMF
from .gmfppp import GMFPPP
from .ncfppp import NCFPPP
from .layers import MultilayerPerceptrons
from .utils import freeze_partial_linear_params, l2_regularization

# Done
class NMFPPP(BaseModule):
    """
    Neural Matrix Factorization +++ with seperated user-item SDAEs.

    - gmfppp_hparams (dict): GMF+++ keyword arguments or hyperparameters.
    - ncfppp_hparams (dict): NCF+++ keyword arguments or hyperparameters.
    - shared_embed (str): Model which to share feature embeddings. Default: None.
        - If "gmf+++", GMF+++ and NCF+++ embeddings will be obtained from GMF+++ features embedding layer, NCF+++ features embedding layer will be deleted/ignored.
        - If "ncf+++", GMF+++ and NCF+++ embeddings will be obtained from NCF+++ features embedding layer, GMF+++ features embedding layer will be deleted/ignored.
        - If None, GMF+++ and NCF+++ will perform feature embeddings seperately.
    - shared_sdaes (str): Model which to share user and item latent features. Also appplied on trade-off parameter alpha and beta. Default: None.
        - If "gmf+++", GMF+++ and NCF+++ latent features will be obtained from GMF+++ user and item SDAEs, NCF+++ user and item SDAEs will be deleted/ignored.
        - If "ncf+++", GMF+++ and NCF+++ latent features will be obtained from NCF+++ user and item SDAEs, GMF+++ user and item SDAEs will be deleted/ignored.
        - If None, GMF+++ and NCF+++ will perform user and item latent representation extraction seperately.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 gmfppp_hparams:dict,
                 ncfppp_hparams:dict,
                 shared_embed:str= None,
                 shared_sdaes:str= None,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        assert shared_embed in [None, "gmf+++", "ncf+++"], "`shared_embed` must be either None, 'gmf+++' or 'ncf+++'."
        assert shared_sdaes in [None, "gmf+++", "ncf+++"], "`shared_sdaes` must be either None, 'gmf+++' or 'ncf+++'."

        super().__init__(lr, 0, criterion)
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.shared_embed = shared_embed
        self.shared_sdaes = shared_sdaes

        self.gmfppp = GMFPPP(**gmfppp_hparams)
        self.ncfppp = NCFPPP(**ncfppp_hparams)

        # If `shared_embed` is not None, delete one of the feature embedding layer based on arg value.
        if shared_embed == "gmf+++":
            del self.ncfppp.features_embedding
        elif shared_embed == "ncf+++":     
            del self.gmfppp.features_embedding  

        # If `shared_embed` is not None, delete one of user and item stacked denoising autoencoder based on arg value.
        if shared_sdaes == "gmf+++":
            del self.ncfppp.user_sdae
            del self.ncfppp.item_sdae
        elif shared_sdaes == "ncf+++":     
            del self.gmfppp.user_sdae
            del self.gmfppp.item_sdae

        # Remove GMF output layer
        del self.gmfppp.net

        # Remove NCF output layer
        del self.ncfppp.net.mlp[-1]
        if type(self.ncfppp.net.mlp[-1]) == torch.nn.Linear:
            del self.ncfppp.net.mlp[-1]

        # Add input dim
        input_dim = gmfppp_hparams["embed_dim"]
        input_dim += gmfppp_hparams["user_sdae_hparams"]["hidden_dims"][-1] and gmfppp_hparams["item_sdae_hparams"]["hidden_dims"][-1]
        input_dim += ncfppp_hparams["hidden_dims"][-1]
        self.net = MultilayerPerceptrons(input_dim = input_dim,  output_layer = "relu")

    def fusion(self, x, user_x, item_x):
        if self.shared_embed == "gmf+++":
            # Share embeddings from GMF+++ features embedding layer
            x_embed_gmfppp = x_embed_ncfppp = self.gmfppp.features_embedding(x.int())
        elif self.shared_embed == "ncf+++":
            # Share embeddings from NCF+++ features embedding layer
            x_embed_gmfppp = x_embed_ncfppp = self.ncfppp.features_embedding(x.int())
        else: 
            # Seperate feature embeddings between GMF+++ and NCF+++
            x_embed_gmfppp = self.gmfppp.features_embedding(x.int())
            x_embed_ncfppp = self.ncfppp.features_embedding(x.int())

        if self.shared_sdaes== "gmf+++":
            # Share latent representation from GMF+++ user and item SDAE.
            # Perform user and features reconstruction (add noise during reconstruction)
            user_y = self.gmfppp.user_sdae.forward(user_x)
            item_y = self.gmfppp.item_sdae.forward(item_x) 
            # Extract user and item latent features (remove noise during extraction)
            user_z_gmfppp = user_z_ncfppp = self.gmfppp.user_sdae.encode(user_x, add_noise = False)
            item_z_gmfppp = item_z_ncfppp = self.gmfppp.item_sdae.encode(item_x, add_noise = False) 
        elif self.shared_sdaes == "ncf+++":
            # Share latent representation from NCF+++ user and item SDAE.
            # Perform user and features reconstruction (add noise during reconstruction)
            user_y = self.ncfppp.user_sdae.forward(user_x)
            item_y = self.ncfppp.item_sdae.forward(item_x) 
            # Extract user and item latent features (remove noise during extraction)
            user_z_gmfppp = user_z_ncfppp = self.ncfppp.user_sdae.encode(user_x, add_noise = False)
            item_z_gmfppp = item_z_ncfppp = self.ncfppp.item_sdae.encode(item_x, add_noise = False) 
        else: 
            # Seperate latent representation extraction between GMF+++ and NCF+++.
            # Perform user and features reconstruction (add noise during reconstruction)
            user_y_gmfppp = self.gmfppp.user_sdae.forward(user_x)
            item_y_gmfppp = self.gmfppp.item_sdae.forward(item_x) 
            user_y_ncfppp = self.ncfppp.user_sdae.forward(user_x)
            item_y_ncfppp = self.ncfppp.item_sdae.forward(item_x) 
            # Extract user and item latent features (remove noise during extraction)
            user_z_gmfppp = self.gmfppp.user_sdae.encode(user_x, add_noise = False)
            item_z_gmfppp = self.gmfppp.item_sdae.encode(item_x, add_noise = False) 
            user_z_ncfppp = self.ncfppp.user_sdae.encode(user_x, add_noise = False)
            item_z_ncfppp = self.ncfppp.item_sdae.encode(item_x, add_noise = False) 
        
        # GMF+++ interaction function
        # Element wise product between user and item embeddings
        user_embed_gmfppp, item_embed_gmfppp = x_embed_gmfppp[:, 0], x_embed_gmfppp[:, 1]
        embed_concat_gmfppp = user_embed_gmfppp * item_embed_gmfppp
        # Element wise product between user and item latent features
        z_concat_gmfppp = user_z_gmfppp * item_z_gmfppp
        # Concatenate user and item embeddings and latent features
        """
        IMPORTANT: Concatenate latent representation 1st, then embeddings.
        So that it is easier to replace the defined embedding weights with pretrained embedding weights. Go to line 119.
        """
        output_gmfppp = torch.cat([z_concat_gmfppp, embed_concat_gmfppp], dim = 1)

        # NCF+++ interaction function
        # Concatenate user and item embeddings
        embed_concat_ncfppp = torch.flatten(x_embed_ncfppp, start_dim = 1)
        # Concatenate user and item latent features.
        z_concat_ncfppp = torch.cat([user_z_ncfppp, item_z_ncfppp], dim = 1)
        # Concatenate user and item emebddings and latent features
        """
        IMPORTANT: Concatenate embeddings 1st, then latent representation.
        So that we can freeze the 1st N weights (embeddings weights), and train the latter weights for latent representation. Go to line 226.
        """
        concat_ncfppp = torch.cat([embed_concat_ncfppp, z_concat_ncfppp], dim = 1)
        # Non linear on concatenated vectors
        output_ncfppp = self.ncfppp.net(concat_ncfppp)
        
        # Concatenate GMF+++ element wise product and NCF+++ last hidden layer output
        fusion = torch.cat([output_ncfppp, output_gmfppp], dim = 1)

        if self.shared_sdaes== "gmf+++":
            return [fusion, user_y, item_y]
        elif self.shared_sdaes == "ncf+++":
            return [fusion, user_y, item_y]
        else:
            return [fusion, user_y_gmfppp, item_y_gmfppp, user_y_ncfppp, item_y_ncfppp]

    def forward(self, x, user_x, item_x):
        # Fusion of GMF+++ and NCF+++
        outputs = self.fusion(x, user_x, item_x)

        # Non linear on fusion vectors
        outputs[0] = self.net(outputs[0])
        
        return outputs

    def l2_regularization(self):
        """
        Total L2 regularization for backward propagration.
        """
        if self.shared_embed == "gmf+++":
            # L2 regularization on GMF+++ feature embeddings layer.
            features_reg = l2_regularization(self.gmfppp.weight_decay, self.gmfppp.features_embedding.parameters(), self.device)
        elif self.shared_embed == "ncf+++":
            # L2 regularization on NCF+++ feature embeddings layer,
            features_reg = l2_regularization(self.ncfppp.weight_decay, self.ncfppp.features_embedding.parameters(), self.device)
        else: 
            # L2 regularization on both GMF+++ and NCF+++ feature embeddings layer.
            features_reg = l2_regularization(self.gmfppp.weight_decay, self.gmfppp.features_embedding.parameters(), self.device)
            features_reg += l2_regularization(self.ncfppp.weight_decay, self.ncfppp.features_embedding.parameters(), self.device)
        # L2 regularization on MLP+++ layer and NeuMF layer.
        ncfppp_reg = l2_regularization(self.weight_decay, self.ncfppp.net.parameters(), self.device)
        nmf_reg = l2_regularization(self.weight_decay, self.net.parameters(), self.device)

        return features_reg + ncfppp_reg + nmf_reg

    def backward_loss(self, *batch):
        """
        Custom backward loss.
        """
        x, user_x, item_x, y = batch

        if self.shared_sdaes== "gmf+++":
            # Prediction
            y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.gmfppp.reconstruction_loss(user_x, item_x, user_x_hat, item_x_hat)
        elif self.shared_sdaes == "ncf+++":
            # Prediction
            y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.ncfppp.reconstruction_loss(user_x, item_x, user_x_hat, item_x_hat)
        else:
            # Prediction
            y_hat, user_x_hat_gmfppp, item_x_hat_gmfppp, user_x_hat_ncfppp, item_x_hat_ncfppp = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.gmfppp.reconstruction_loss(user_x, item_x, user_x_hat_gmfppp, item_x_hat_gmfppp)
            reconstruction_loss += self.ncfppp.reconstruction_loss(user_x, item_x, user_x_hat_ncfppp, item_x_hat_ncfppp)

        # Rating loss
        rating_loss = self.criterion(y_hat, y)
        
        # L2 regularization
        reg = self.l2_regularization()

        # Total loss
        loss = rating_loss + reconstruction_loss + reg
        
        return loss

    def logger_loss(self, *batch):
        """
        Overwrite logger loss which focus evaluation on y_hat only
        """
        xs = batch[:-1]
        y = batch[-1]
        preds = self.forward(*xs)
        loss = self.criterion(preds[0], y)

        return loss

    def load_all_pretrain_weights(self, gmfppp_weights, ncfppp_weights, freeze = True):
        """
        Load pretrain weights for GMF+++ and NCF+++.
        """
        gmfppp_weights = {k:v for k,v in gmfppp_weights.items() if k in self.gmfppp.state_dict()}
        ncfppp_weights = {k:v for k,v in ncfppp_weights.items() if k in self.ncfppp.state_dict()}

        self.gmfppp.load_state_dict(gmfppp_weights)
        self.ncfppp.load_state_dict(ncfppp_weights)

        if freeze:
            self.gmfppp.freeze()
            self.ncfppp.freeze()

    def load_nmf_pretrain_weights(self, nmf_weights, freeze = True):
        """
        Load pretrain weights for NeuMF layer.
        """
        nmfppp_weights = self.state_dict()
        # Rename keys which contains "gmf" and "ncf" by adding "ppp" to them
        nmf_weights = {k.replace("f.", "fppp.") if "gmf" in k or "ncf" in k
                       else k:nmf_weights[k] for k in nmf_weights.keys()}

        nmf_input_weights = nmf_weights["net.mlp.0.weight"]
        nmf_input_dim = nmf_input_weights.shape[-1]
        nmf_weights["net.mlp.0.weight"] = nmfppp_weights["net.mlp.0.weight"]
        nmf_weights["net.mlp.0.weight"][:, -nmf_input_dim:] = nmf_input_weights
        
        ncf_input_weights = nmf_weights["ncfppp.net.mlp.0.weight"] 
        ncf_input_dim = ncf_input_weights.shape[-1]
        nmf_weights["ncfppp.net.mlp.0.weight"] = nmfppp_weights["ncfppp.net.mlp.0.weight"]
        nmf_weights["ncfppp.net.mlp.0.weight"][:, :ncf_input_dim] = ncf_input_weights

        nmfppp_weights.update(nmf_weights)
        self.load_state_dict(nmfppp_weights)

        if freeze:
            # manually freeze layer using require_grad = False, without using pytorch lightning freeze method
            for name, param in self.named_parameters():
                if  name in nmf_weights.keys():
                    # Freeze neumf layers except for
                    if name not in ["ncfppp.net.mlp.0.weight", # NCF input layer weights
                                    "ncfppp.net.mlp.0.bias", # NCF input layer biases
                                    "net.mlp.0.weight", # NeuMF layer weights
                                    "net.mlp.0.bias"]: # NeuMF layer biases
                        param.requires_grad = False

            # Use this code for checking
            """for name, param in self.named_parameters():
                print(name, param.requires_grad)"""

            # Freeze the first `ncf_input_dim` weights as the latter weights are for newly added concatenated latent representation
            freeze_partial_linear_params(self.ncfppp.net.mlp[0], list(range(ncf_input_dim)), dim = 1)