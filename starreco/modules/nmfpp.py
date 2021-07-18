import torch
import torch.nn.functional as F

from .module import BaseModule
from .nmf import NMF
from .gmfpp import GMFPP
from .ncfpp import NCFPP
from .layers import MultilayerPerceptrons
from .utils import freeze_partial_linear_params, l2_regularization

# Done
class NMFPP(BaseModule):
    """
    Neural Matrix Factorization ++ with seperated user-item SDAEs.

    - gmfpp_hparams (dict): GMF++ keyword arguments or hyperparameters.
    - ncfpp_hparams (dict): NCF++ keyword arguments or hyperparameters.
    - shared_embed (str): Model which to share feature embeddings. Default: None.
        - If "gmf++", GMF++ and NCF++ embeddings will be obtained from GMF++ features embedding layer, NCF++ features embedding layer will be deleted/ignored.
        - If "ncf++", GMF++ and NCF++ embeddings will be obtained from NCF++ features embedding layer, GMF++ features embedding layer will be deleted/ignored.
        - If None, GMF++ and NCF++ will perform feature embeddings seperately.
    - shared_sdaes (str): Model which to share user and item latent features. Also applied on trade-off parameter alpha and beta. Default: None.
        - If "gmf++", GMF++ and NCF++ latent features will be obtained from GMF++ user and item SDAEs, NCF++ user and item SDAEs will be deleted/ignored.
        - If "ncf++", GMF++ and NCF++ latent features will be obtained from NCF++ user and item SDAEs, GMF++ user and item SDAEs will be deleted/ignored.
        - If None, GMF++ and NCF++ will perform user and item latent representation extraction seperately.
    - lr (float): Learning rate. Default: 1e-3.
    - weight_decay (float): L2 regularization rate. Default: 1e-3.
    - criterion: Criterion or objective or loss function. Default: F.mse_loss.
    """

    def __init__(self, 
                 gmfpp_hparams:dict,
                 ncfpp_hparams:dict,
                 shared_embed:str= None,
                 shared_sdaes:str= None,
                 lr:float = 1e-3,
                 weight_decay:float = 1e-3,
                 criterion = F.mse_loss):
        assert shared_embed in [None, "gmf++", "ncf++"], "`shared_embed` must be either None, 'gmf++' or 'ncf++'."
        assert shared_sdaes in [None, "gmf++", "ncf++"], "`shared_sdaes` must be either None, 'gmf++' or 'ncf++'."

        super().__init__(lr, 0, criterion)
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.shared_embed = shared_embed
        self.shared_sdaes = shared_sdaes

        self.gmfpp = GMFPP(**gmfpp_hparams)
        self.ncfpp = NCFPP(**ncfpp_hparams)

        # If `shared_embed` is not None, delete one of the feature embedding layer based on arg value.
        if shared_embed == "gmf++":
            del self.ncfpp.features_embedding
        elif shared_embed == "ncf++":     
            del self.gmfpp.features_embedding  

        # If `shared_embed` is not None, delete one of user and item stacked denoising autoencoder based on arg value.
        if shared_sdaes == "gmf++":
            del self.ncfpp.user_sdae
            del self.ncfpp.item_sdae
        elif shared_sdaes == "ncf++":     
            del self.gmfpp.user_sdae
            del self.gmfpp.item_sdae

        # Remove GMF output layer
        del self.gmfpp.net

        # Remove NCF output layer
        del self.ncfpp.net.mlp[-1]
        if type(self.ncfpp.net.mlp[-1]) == torch.nn.Linear:
            del self.ncfpp.net.mlp[-1]

        # Add input dim
        input_dim = gmfpp_hparams["embed_dim"]
        input_dim += gmfpp_hparams["user_sdae_hparams"]["hidden_dims"][-1] and gmfpp_hparams["item_sdae_hparams"]["hidden_dims"][-1]
        input_dim += ncfpp_hparams["hidden_dims"][-1]
        self.net = MultilayerPerceptrons(input_dim = input_dim,  output_layer = "relu")

    def forward(self, x, user_x, item_x):
        # Fusion of GMF++ and NCF++
        if self.shared_embed == "gmf++":
            # Share embeddings from GMF++ features embedding layer
            x_embed_gmfpp = x_embed_ncfpp = self.gmfpp.features_embedding(x.int())
        elif self.shared_embed == "ncf++":
            # Share embeddings from NCF++ features embedding layer
            x_embed_gmfpp = x_embed_ncfpp = self.ncfpp.features_embedding(x.int())
        else: 
            # Seperate feature embeddings between GMF++ and NCF++
            x_embed_gmfpp = self.gmfpp.features_embedding(x.int())
            x_embed_ncfpp = self.ncfpp.features_embedding(x.int())

        if self.shared_sdaes== "gmf++":
            # Share latent representation from GMF++ user and item SDAE.
            # Perform user and features reconstruction (add noise during reconstruction)
            user_y = self.gmfpp.user_sdae.forward(user_x)
            item_y = self.gmfpp.item_sdae.forward(item_x) 
            # Extract user and item latent features (remove noise during extraction)
            user_z_gmfpp = user_z_ncfpp = self.gmfpp.user_sdae.encode(user_x, add_noise = False)
            item_z_gmfpp = item_z_ncfpp = self.gmfpp.item_sdae.encode(item_x, add_noise = False) 
        elif self.shared_sdaes == "ncf++":
            # Share latent representation from NCF++ user and item SDAE.
            # Perform user and features reconstruction (add noise during reconstruction)
            user_y = self.ncfpp.user_sdae.forward(user_x)
            item_y = self.ncfpp.item_sdae.forward(item_x) 
            # Extract user and item latent features (remove noise during extraction)
            user_z_gmfpp = user_z_ncfpp = self.ncfpp.user_sdae.encode(user_x, add_noise = False)
            item_z_gmfpp = item_z_ncfpp = self.ncfpp.item_sdae.encode(item_x, add_noise = False) 
        else: 
            # Seperate latent representation extraction between GMF++ and NCF++.
            # Perform user and features reconstruction (add noise during reconstruction)
            user_y_gmfpp = self.gmfpp.user_sdae.forward(user_x)
            item_y_gmfpp = self.gmfpp.item_sdae.forward(item_x) 
            user_y_ncfpp = self.ncfpp.user_sdae.forward(user_x)
            item_y_ncfpp = self.ncfpp.item_sdae.forward(item_x) 
            # Extract user and item latent features (remove noise during extraction)
            user_z_gmfpp = self.gmfpp.user_sdae.encode(user_x, add_noise = False)
            item_z_gmfpp = self.gmfpp.item_sdae.encode(item_x, add_noise = False) 
            user_z_ncfpp = self.ncfpp.user_sdae.encode(user_x, add_noise = False)
            item_z_ncfpp = self.ncfpp.item_sdae.encode(item_x, add_noise = False) 
        
        # GMF++ interaction function
        # Element wise product between user and item embeddings
        user_embed_gmfpp, item_embed_gmfpp = x_embed_gmfpp[:, 0], x_embed_gmfpp[:, 1]
        embed_concat_gmfpp = user_embed_gmfpp * item_embed_gmfpp
        # Element wise product between user and item latent features
        z_concat_gmfpp = user_z_gmfpp * item_z_gmfpp
        # Concatenate user and item embeddings and latent features
        """
        IMPORTANT: Concatenate latent representation 1st, then embeddings.
        So that it is easier to replace the defined embedding weights with pretrained embedding weights. Go to line 119.
        """
        output_gmfpp = torch.cat([z_concat_gmfpp, embed_concat_gmfpp], dim = 1)

        # NCF++ interaction function
        # Concatenate user and item embeddings
        embed_concat_ncfpp = torch.flatten(x_embed_ncfpp, start_dim = 1)
        # Concatenate user and item latent features.
        z_concat_ncfpp = torch.cat([user_z_ncfpp, item_z_ncfpp], dim = 1)
        # Concatenate user and item emebddings and latent features
        """
        IMPORTANT: Concatenate embeddings 1st, then latent representation.
        So that we can freeze the 1st N weights (embeddings weights), and train the latter weights for latent representation. Go to line 226.
        """
        concat_ncfpp = torch.cat([embed_concat_ncfpp, z_concat_ncfpp], dim = 1)
        # Non linear on concatenated vectors
        output_ncfpp = self.ncfpp.net(concat_ncfpp)
        
        # Concatenate GMF++ element wise product and NCF++ last hidden layer output
        fusion = torch.cat([output_ncfpp, output_gmfpp], dim = 1)

        # Non linear on fusion vectors
        y = self.net(fusion)

        if self.shared_sdaes== "gmf++":
            return y, user_y, item_y
        elif self.shared_sdaes == "ncf++":
            return y, user_y, item_y
        else:
            return y, user_y_gmfpp, item_y_gmfpp, user_y_ncfpp, item_y_ncfpp

    def l2_regularization(self):
        """
        Total L2 regularization for backward propagration.
        """
        if self.shared_embed == "gmf++":
            # L2 regularization on GMF++ feature embeddings layer.
            features_reg = l2_regularization(self.gmfpp.weight_decay, self.gmfpp.features_embedding.parameters(), self.device)
        elif self.shared_embed == "ncf++":
            # L2 regularization on NCF++ feature embeddings layer,
            features_reg = l2_regularization(self.ncfpp.weight_decay, self.ncfpp.features_embedding.parameters(), self.device)
        else: 
            # L2 regularization on both GMF++ and NCF++ feature embeddings layer.
            features_reg = l2_regularization(self.gmfpp.weight_decay, self.gmfpp.features_embedding.parameters(), self.device)
            features_reg += l2_regularization(self.ncfpp.weight_decay, self.ncfpp.features_embedding.parameters(), self.device)
        # L2 regularization on MLP++ layer and NeuMF layer.
        ncfpp_reg = l2_regularization(self.weight_decay, self.ncfpp.net.parameters(), self.device)
        nmf_reg = l2_regularization(self.weight_decay, self.net.parameters(), self.device)

        return features_reg + ncfpp_reg + nmf_reg

    def backward_loss(self, *batch):
        """
        Custom backward loss.
        """
        x, user_x, item_x, y = batch

        if self.shared_sdaes== "gmf++":
            # Prediction
            y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.gmfpp.reconstruction_loss(user_x, item_x, user_x_hat, item_x_hat)
        elif self.shared_sdaes == "ncf++":
            # Prediction
            y_hat, user_x_hat, item_x_hat = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.ncfpp.reconstruction_loss(user_x, item_x, user_x_hat, item_x_hat)
        else:
            # Prediction
            y_hat, user_x_hat_gmfpp, item_x_hat_gmfpp, user_x_hat_ncfpp, item_x_hat_ncfpp = self.forward(x, user_x, item_x)
            # Reconstruction loss
            reconstruction_loss = self.gmfpp.reconstruction_loss(user_x, item_x, user_x_hat_gmfpp, item_x_hat_gmfpp)
            reconstruction_loss += self.ncfpp.reconstruction_loss(user_x, item_x, user_x_hat_ncfpp, item_x_hat_ncfpp)

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

    def load_all_pretrain_weights(self, gmfpp_weights, ncfpp_weights, freeze = True):
        """
        Load pretrain weights for GMF++ and NCF++.
        """
        gmfpp_weights = {k:v for k,v in gmfpp_weights.items() if k in self.gmfpp.state_dict()}
        ncfpp_weights = {k:v for k,v in ncfpp_weights.items() if k in self.ncfpp.state_dict()}

        self.gmfpp.load_state_dict(gmfpp_weights)
        self.ncfpp.load_state_dict(ncfpp_weights)

        if freeze:
            self.gmfpp.freeze()
            self.ncfpp.freeze()

    def load_nmf_pretrain_weights(self, nmf_weights, freeze = True):
        """
        Load pretrain weights for NeuMF layer.
        """
        nmfpp_weights = self.state_dict()
        # Rename keys which contains "gmf" and "ncf" by adding "pp" to them
        nmf_weights = {k.replace("f.", "fpp.") if "gmf" in k or "ncf" in k
                       else k:nmf_weights[k] for k in nmf_weights.keys()}

        nmf_input_weights = nmf_weights["net.mlp.0.weight"]
        nmf_input_dim = nmf_input_weights.shape[-1]
        nmf_weights["net.mlp.0.weight"] = nmfpp_weights["net.mlp.0.weight"]
        nmf_weights["net.mlp.0.weight"][:, -nmf_input_dim:] = nmf_input_weights
        
        ncf_input_weights = nmf_weights["ncfpp.net.mlp.0.weight"] 
        ncf_input_dim = ncf_input_weights.shape[-1]
        nmf_weights["ncfpp.net.mlp.0.weight"] = nmfpp_weights["ncfpp.net.mlp.0.weight"]
        nmf_weights["ncfpp.net.mlp.0.weight"][:, :ncf_input_dim] = ncf_input_weights

        nmfpp_weights.update(nmf_weights)
        self.load_state_dict(nmfpp_weights)

        if freeze:
            # manually freeze layer using require_grad = False, without using pytorch lightning freeze method
            for name, param in self.named_parameters():
                if  name in nmf_weights.keys():
                    # Freeze neumf layers except for
                    if name not in ["ncfpp.net.mlp.0.weight", # NCF input layer weights
                                    "ncfpp.net.mlp.0.bias", # NCF input layer biases
                                    "net.mlp.0.weight", # NeuMF layer weights
                                    "net.mlp.0.bias"]: # NeuMF layer biases
                        param.requires_grad = False

            # Use this code for checking
            """for name, param in self.named_parameters():
                print(name, param.requires_grad)"""

            # Freeze the first `ncf_input_dim` weights as the latter weights are for newly added concatenated latent representation
            freeze_partial_linear_params(self.ncfpp.net.mlp[0], list(range(ncf_input_dim)), dim = 1)