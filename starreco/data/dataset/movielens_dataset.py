import zipfile
import requests
import time
import warnings
import os
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

from .dataset import BaseDataset, User, Item, Rating

# Done
class MovielensDataset(BaseDataset):
    """
    Dataset class for Movielens.
    """

    def import_data(self):
        """
        Import Movielens rating, user and movie dataframes. 
        """
        dataset_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-1m.zip")
        zf = zipfile.ZipFile(dataset_path)

        # User dataframe
        users = pd.read_csv(zf.open(f"ml-1m/users.dat"), delimiter = "::", names = ["userId", "gender", "age", "occupation", "zipCode"], engine = "python")
        user = User(users, "userId")
        user.cat_columns = ["gender", "occupation", "age", "zipCode"]

        # Item dataframe
        movies = pd.read_csv(zf.open(f"ml-1m/movies.dat"), delimiter = "::", names = ["movieId", "title", "genre"], encoding = "ISO-8859-1", engine = "python")
        # Get movies plot
        if self.crawl_data:
            movies = self.get_movies_with_imdbs(movies)
            movies = self.get_movies_with_plots(movies)
        movies["genre"] = movies["genre"].apply(lambda x:set(x.split("|")))
        item = Item(movies, "movieId")
        item.set_columns = ["genre"]
        if self.crawl_data:
            item.doc_columns = ["plot"]

        # Rating dataframe
        ratings = pd.read_csv(zf.open(f"ml-1m/ratings.dat"), delimiter = "::", 
                              names = ["userId", "movieId", "rating", "timestamp"], engine = "python")
        rating = Rating(ratings, "rating", user, item)

        return rating, user, item

    def get_movies_with_imdbs(self, movies_1m):
        """
        Get movies with plots.
        """
        movielinks_extra_path = "ml-1m_links.csv"

        # Merge movielen-25m movies.csv and links.csv
        ml_25m_path = super().download_data(f"http://files.grouplens.org/datasets/movielens/ml-25m.zip")
        ml_25m = zipfile.ZipFile(ml_25m_path)
        movies_25m = pd.read_csv(ml_25m.open(f"ml-25m/movies.csv"), encoding = "ISO-8859-1", engine = "python")
        links_25m = pd.read_csv(ml_25m.open(f"ml-25m/links.csv"), encoding = "ISO-8859-1", engine = "python")
        movielinks_25m = movies_25m.merge(links_25m, on = "movieId")

        # Uncomment the code below for testing.
        # Use the export csv from the code below to check between alternative titles.
        """
        import difflib
        movielinks_25m = movielinks_25m[["movieId", "title", "imdbId"]]
        movies = movies_1m.merge(movielinks_25m, on = ["movieId"], how = "left")
        temp = movies[movies["title_x"] != movies["title_y"]][~movies["imdbId"].isna()][["title_x","title_y"]]
        # Using python difflib library to find the similarity between titles
        temp["similarity"] = temp.apply(lambda row: (lambda seq = 
                                        difflib.SequenceMatcher(None, row["title_x"], row["title_y"]):
                                        "%.2f" % seq.quick_ratio())(),
                                        axis = 1)
        temp = temp.sort_values("similarity")
        temp.to_csv("movies_25m_1m_title_check.csv", index = False)
        """

        # Uncomment the code below for testing.
        # Use the export csv from the code below to manually insert imdbId.
        """
        temp = movies[movies["imdbId"].isna()]
        temp.to_csv("movies_25m_1m_empty_links.csv", index = False)
        """

        # Merge movielens-1m movies and movielens-25m movielinks
        movielinks_25m = movielinks_25m[["movieId", "imdbId"]]

        # Some IMDB_ID are missed in the movielens-25m, so add manually
        if os.path.isfile(movielinks_extra_path):
            movielinks_extra = pd.read_csv(movielinks_extra_path, encoding = "ISO-8859-1", engine = "python")
            movies = movies_1m.merge(movielinks_25m, on = "movieId", how = "left")
            movies.update(movies[["movieId"]].merge(movielinks_extra, on = ["movieId"], how = "left"))

        return movies   

    def get_movies_with_plots(self, movies = None):
        """
        Crawl movie plots from IMDB website.
        """
        movies_path = "ml-1m-movies.csv"
        last_valid_index = 0

        # Create a logging file using current timestamp.
        timestamp = int(time.time())
        logging.basicConfig(filename = f"movies_logging_{timestamp}.log",
                            format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt = '%Y-%m-%d:%H:%M:%S',
                            level = logging.INFO)
        print(f"Log file created in your working directory at {timestamp}")

        # If path exist, resume from last description valid index
        if os.path.isfile(movies_path):            
            movies = pd.read_csv(movies_path, encoding = "ISO-8859-1", engine = "python")
            last_valid_index = movies["plot"].last_valid_index() + 1

            print(f"{movies_path} exist. Continue from last description value index {last_valid_index}.")
            logging.info(f"{movies_path} exist. Continue from last description value index {last_valid_index}.")
        else:
            assert movies is not None

        # Start crawling
        print("Start crawling")
        for i in tqdm(range(last_valid_index, len(movies)), total = len(movies)):
            movie_id = movies.loc[i, "movieId"]
            imdb_id = movies.loc[i, "imdbId"]
            plot = np.nan
            if not pd.isna(imdb_id):
                attempt = 0
                while attempt < 10: # maximum 10 attempts
                    try:
                        # Plot crawling from IMDB site
                        imdb_link = f"https://www.imdb.com/title/tt{str(int(imdb_id)).zfill(7)}"
                        response = requests.get(imdb_link)
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Find plot from soup
                        plot = soup.find_all("div", class_="Storyline__StorylineWrapper-sc-1b58ttw-0 iywpty")[0].div
                        if plot.find("span") is not None:
                            plot.span.decompose()
                        plot = plot.text

                        if plot: 
                            logging.info(f"Success at movieId {movie_id}.")
                        else:
                            logging.warning(f"Success with empty plot at movieId {movie_id}")
                        break   
                    except Exception as e:
                        attempt += 1
                        warnings.warn(f"Reattempt crawling at movieId {movie_id} for the {attempt}th time(s). {e}")
                        logging.warning(f"Reattempt crawling at movieId {movie_id} for the {attempt}th time(s). {e}")
                        time.sleep(5) 

                        if attempt == 10:
                            warnings.warn(f"Fail at movieId {movie_id}. Please check your logging file.")
                            logging.error(f"Fail at movieId {movie_id}.")
            else:
                # If imdbId is empty, proceed to the next movieId and provide a warning log.    
                logging.warning(f"Empty imdbId at movieId {movie_id}")   

            # Insert plot and export dataframe
            movies.loc[i, "plot"] = plot
            movies.to_csv(movies_path, index = False)
        print("End crawling")  
        return movies     

