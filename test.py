from starreco.data import *

dataset = "book-crossing"

for Class in [
    DataModule(dataset), HybridDataModule(dataset), AEDataModule(dataset), 
    HybridAEDataModule(dataset, transpose = True), HybridAEDataModule(dataset, transpose = False)
]:
    data = Class
    data.setup()
    data.train_dataloader()
    data.val_dataloader()
    data.test_dataloader()
    print(data)