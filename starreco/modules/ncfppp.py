from typing import Union

from . import NCFPP
from .utils import l2_regularization

class NCFPPP(NCFPP):
    def __init__(self, 
                 user_criterion_alpha:Union[int,float] = 1,
                 user_criterion_beta:Union[int,float] = 1,
                 item_criterion_alpha:Union[int,float] = 1,
                 item_criterion_beta:Union[int,float] = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_criterion_alpha = user_criterion_alpha
        self.user_criterion_beta = user_criterion_beta
        self.item_criterion_alpha = item_criterion_alpha
        self.item_criterion_beta = item_criterion_beta

    def reconstruction_loss(self, user_x, item_x, user_x_hat, item_x_hat):
        """
        Reconstruction loss.
        """
        # User denoising and reconstruction loss
        # Get noise mask
        user_noise_mask = self.user_sdae.encoder_noise_masks[0]
        # Denoising loss
        user_x_denoised = user_x * user_noise_mask
        user_x_hat_denoised = user_x_hat * user_noise_mask
        user_denoising_loss = self.user_criterion_alpha * self.user_criterion(user_x_hat_denoised, user_x_denoised)
        # Reconstruction loss
        user_x_reconstructed = user_x * ~user_noise_mask
        user_x_hat_reconstructed = user_x_hat * ~user_noise_mask
        user_reconstruction_loss = self.user_criterion_beta * self.user_criterion(user_x_hat_reconstructed, user_x_reconstructed)
        user_loss = user_denoising_loss + user_reconstruction_loss
        user_reg = l2_regularization(self.user_weight_decay, self.user_sdae.parameters(), self.device)
        user_loss *= self.criterion_alpha
        user_loss += user_reg

        # Item denoising and reconstruction loss
        # Get noise mask
        item_noise_mask = self.item_sdae.encoder_noise_masks[0]
        # Denoising loss
        item_x_denoised = item_x * item_noise_mask
        item_x_hat_denoised = item_x_hat * item_noise_mask
        item_denoising_loss = self.item_criterion_alpha * self.item_criterion(item_x_hat_denoised, item_x_denoised)
        # Reconstruction loss
        item_x_reconstructed = item_x * ~item_noise_mask
        item_x_hat_reconstructed = item_x_hat * ~item_noise_mask
        item_reconstruction_loss = self.item_criterion_beta * self.item_criterion(item_x_hat_reconstructed, item_x_reconstructed)
        item_loss = item_denoising_loss + item_reconstruction_loss
        item_reg = l2_regularization(self.item_weight_decay, self.item_sdae.parameters(), self.device)
        item_loss *= self.criterion_alpha
        item_loss += item_reg
        
        return user_loss + item_loss