from starreco.data import *

dataset = "ml-1m"

for Class in [
    DataModule(dataset), HybridDataModule(dataset), 
    AEDataModule(dataset), AEDataModule(dataset, transpose = True),
    HybridAEDataModule(dataset), HybridAEDataModule(dataset, transpose = True)
]:
    data = Class
    data.setup()
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()