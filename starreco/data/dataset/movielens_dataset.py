import zipfile

import pandas as pd

from starreco.data.dataset.dataset import Dataset
from starreco.data.dataset.dataframe import User, Item, Rating

class MovielensDataset(Dataset):
    def import_data(self):
        """
        Import Movielens dataset
        ~ Movielens also provides latest dataset, but only suitable for development and eduction.
        ~ Latest Movielens dataset also not appropriate for reporting research results.
        ~ Hence, old datasets are utilized for benchmarking.
        ~ For more info: https://grouplens.org/datasets/movielens/
        """
        dataset_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        zf = zipfile.ZipFile(dataset_path)

        user = pd.read_csv(zf.open(f"ml-1m/users.dat"), delimiter = "::",
        names = ["userId", "gender", "age", "occupation", "zipCode"], engine = "python")
        user = User(user, "userId")
        user.cat_columns = ["gender", "occupation", "zipCode"]
        user.num_columns = ["age"]

        item = pd.read_csv(zf.open(f"ml-1m/movies.dat"), delimiter = "::",
        names = ["movieId", "title", "genre"], encoding = "ISO-8859-1", engine = "python")
        item["genre"] = item["genre"].apply(lambda x:set(x.split("|")))
        item = Item(item, "movieId")
        item.set_columns = ["genre"]

        rating = pd.read_csv(zf.open(f"ml-1m/ratings.dat"), delimiter = "::", 
        names = ["userId", "movieId", "rating", "timestamp"], engine = "python")
        rating = Rating(rating, user, item , "rating")

        return rating, user, item


    def __init__(self, size = "1m"):
        self.size = size # ** haven't implement for various movielens dataset size **
        super().__init__()