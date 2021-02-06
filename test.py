from starreco.data import DataModule
dataset = DataModule("ml-1m")
dataset.prepare_data()