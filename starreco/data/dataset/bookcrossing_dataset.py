import zipfile
import requests
import time
import warnings
import logging


import pandas as pd
from tqdm import tqdm

from .dataset import BaseDataset, User, Item, Rating


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

        # Read csv
        ratings = pd.read_csv(zf.open("BX-Book-Ratings.csv"), delimiter = ";", escapechar = "\\", encoding = "ISO-8859-1")
        users = pd.read_csv(zf.open("BX-Users.csv"), delimiter = ";", escapechar = "\\", encoding = "ISO-8859-1")
        books = pd.read_csv(zf.open("BX-Books.csv"), delimiter = ";", escapechar = "\\", encoding = "ISO-8859-1")
        
        # User dataframe
        user = User(users, "User-ID")
        user.cat_columns = ["Location"]
        user.num_columns = ["Age"]
        
        # Item dataframe
        books = self.get_books_with_descriptions(books)
        item = Item(books, "ISBN")
        item.cat_columns = ["Book-Author", "Publisher"]

        # Rating dataframe
        rating = Rating(ratings, "Book-Rating", user, item)

        return rating, user, item

    def get_books_with_descriptions(self, books):
        books_path = "BX-Books.csv"

        # Create a logging file using current timestamp.
        timestamp = int(time.time())
        logging.basicConfig(filename = f"bx_logging_{timestamp}.log",
                            format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt = '%Y-%m-%d:%H:%M:%S',
                            level = logging.INFO)
        print(f"Log file created in your working directory at {timestamp}")

        # Start crawling
        print("Start crawling")
        for i, isbn in tqdm(enumerate(books["ISBN"]), total = len(books)):
            description = ""
            google_api_link = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn.zfill(10)}"

            attempt = 0
            while attempt < 10: # maximum 10 attempts
                response = requests.get(google_api_link)

                if response.status_code == 200:
                    try:
                        description = response.json()["items"][0]["volumeInfo"]["description"]
                        logging.info(f"Success at ISBN {isbn}.")
                    except Exception as e:
                        logging.warning(f"Success with empty description at ISBN {isbn}")
                    break
                else:
                    attempt += 1
                    warnings.warn(f"Reattempt crawling at ISBN {isbn} for the {attempt}th time(s). Status code {response.status_code}")
                    logging.warning(f"Reattempt crawling at ISBN {isbn} for the {attempt}th time(s). Status code {response.status_code}")
                    time.sleep(5) 

                    if attempt == 10:
                        warnings.warn(f"Fail at ISBN {isbn}. Please check your logging file.")
                        logging.error(f"Fail at ISBN {isbn}.")

            # Insert description and export dataframe
            books.loc[i, "description"] = description
            books.to_csv(books_path, index = False)

        print("End crawling")
        return books

