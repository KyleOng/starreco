import zipfile

import pandas as pd

from ._dataset import BaseDataset, User, Item, Rating

class MovielensDataset(BaseDataset):
    """
    Dataset class for Movielens.
    """

    def import_data(self):
        """
        Import Movielens rating, user and item dataframes. 
        """

        # Download dataset from url and store as zipfile
        dataset_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        zf = zipfile.ZipFile(dataset_path)

        # User dataframe
        user = pd.read_csv(zf.open(f"ml-1m/users.dat"), delimiter = "::",
                           names = ["userId", "gender", "age", "occupation", "zipCode"], engine = "python")
        user = User(user, "userId")
        user.cat_columns = ["gender", "occupation", "age"]

        # Item dataframe
        item = pd.read_csv(zf.open(f"ml-1m/movies.dat"), delimiter = "::",
                           names = ["movieId", "title", "genre"], encoding = "ISO-8859-1", engine = "python")
        item["genre"] = item["genre"].apply(lambda x:set(x.split("|")))
        item = Item(item, "movieId")
        item.set_columns = ["genre"]

        # Rating dataframe
        rating = pd.read_csv(zf.open(f"ml-1m/ratings.dat"), delimiter = "::", 
                             names = ["userId", "movieId", "rating", "timestamp"], engine = "python")
        rating = Rating(rating, "rating", user, item)

        return rating, user, item