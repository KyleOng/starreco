from starreco.data import *

dataset = "epinions"

data = RCSIDataModule(dataset)
data.setup()
data.train_dataloader()
data.val_dataloader()
data.test_dataloader()