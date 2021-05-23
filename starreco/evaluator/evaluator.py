import math

import torch.nn.functional as F
from tqdm import tqdm

# To do
class RMSEEvaluator:
    """
    Root Mean Squared Error (RMSE) Evaluator

    - m (model): Recommendation model.
    - dataloader (list): Data loader.
    """

    @staticmethod
    def evaluate(m, dataloader):
        infer_m = m.eval()
        total_rmse = 0
        batch_total = 0
        for batch in tqdm(dataloader):
            xs = batch[:-1]
            y = batch[-1]
            y_hat = infer_m.forward(*xs)
            mse = F.mse_loss(y, y_hat)
            total_rmse += math.sqrt(mse)
            batch_total += 1
        return total_rmse/batch_total