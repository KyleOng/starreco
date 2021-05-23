import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Done
class MSEEvaluator:
    """
    Mean Squared Error (MSE) Evaluator

    - m (model): Recommendation model.
    - dataloader (list): Data loader.
    """

    @staticmethod
    def evaluate(m, dataloader):
        m_infer = m.eval()
        mses = torch.tensor([])
        for batch in tqdm(dataloader):
            xs = batch[:-1]
            y = batch[-1]
            y_hat = m_infer.forward(*xs)
            mse = F.mse_loss(y, y_hat).unsqueeze(0)
            mses = torch.cat([mses, mse])
        return torch.mean(mses).item()

# Done
class RMSEEvaluator(MSEEvaluator):
    """
    Root Mean Squared Error (RMSE) Evaluator

    - m (model): Recommendation model.
    - dataloader (list): Data loader.
    """

    @staticmethod
    def evaluate(m, dataloader):
        return math.sqrt(super().evaluate(m, dataloader))