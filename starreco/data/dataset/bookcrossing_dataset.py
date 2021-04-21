import zipfile

import pandas as pd

from ._dataframe import User, Item, Rating
from ._dataset import BaseDataset

class BookCrossingDataset(BaseDataset):
    """
    Dataset class for Book Crossing.
    """

    def import_data(self):
        """
        Import Book Crossing rating, user and item dataframes. 
        """

        # Download dataset from url and store as zipfile
        dataset_path = super().download_data("http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip")
        zf = zipfile.ZipFile(dataset_path)

        # User dataframe
        user = pd.read_csv(zf.open("BX-Users.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        user = User(user, "User-ID")
        user.cat_columns = ["Location"]
        user.num_columns = ["Age"]

        # Item dataframe
        item = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        item = Item(item, "ISBN")
        item.cat_columns = ["Book-Author", "Publisher"]

        # Rating dataframe
        rating = pd.read_csv(zf.open("BX-Book-Ratings.csv"), delimiter = ";",
        escapechar = "\\", encoding = "ISO-8859-1")
        rating = Rating(rating, user, item, "Book-Rating")

        return rating, user, item