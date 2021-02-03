from starreco.data import DataModule
dataset = DataModule("book-crossing")
print(dataset.prepare_data())