from starreco.data import *

dataset = "ml-1m"

data = RCSIDataModule(dataset)
data.setup()
for x,y in data.train_dataloader():
    print(x.shape, y.shape)
    break