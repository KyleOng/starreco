import torch

# Done
class FeaturesEmbeddingMixin:
    """Features Embedding Mixin."""

    def user_item_embeddings(self, x:torch.Tensor, concat:bool = False):
        """ 
        Return user and item embeddings

        - x (torch.Tensor): x tensor.
        - concat (bool): If True concat use and item embeddings, else seperate.       
        """
        
        try:
            # Generate embeddings
            x_embed = self.features_embedding(x.int())

            if concat:
                return torch.flatten(x_embed, start_dim = 1)
            else:
                # Seperate user (1st column) and item (2nd column) embeddings from generated embeddings
                user_embed = x_embed[:, 0]
                item_embed = x_embed[:, 1]

                return user_embed, item_embed
        except NameError as e:
            raise f"{e}. Please make sure that `self.features_embedding` is intialize in `self.__init__()`"