import zipfile

import pandas as pd

from starreco.data.dataset.dataset import Dataset
from starreco.data.dataset.dataframe import User, Item, Rating

class BookCrossingDataset(Dataset):
    def import_data(self):
        """
        Import Book Crossing Dataset
        """
        # Get dataset
        dataset_path = super().download_data("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
        zf = zipfile.ZipFile(dataset_path)

        user = pd.read_csv(zf.open("BX-Users.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        user = User(user, "User-ID")
        user.cat_columns = ["Location"]
        user.num_columns = ["Age"]

        item = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        item = Item(item, "ISBN")
        item.cat_columns = ["Book-Author", "Publisher"]

        rating = pd.read_csv(zf.open("BX-Book-Ratings.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        rating = Rating(rating, user, item, "Book-Rating")

        return rating, user, item
    