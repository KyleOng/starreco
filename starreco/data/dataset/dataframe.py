import pandas as pd 
import numpy as np

class Object:
    cat_columns = num_columns = set_columns = []

    def __init__(self, df, column):
        self.df = df if df is not None else pd.DataFrame(columns = [column])
        self.column = column
        
    def map_column(self, map, join = "left"):
        df = self.df.copy()
        self.map_ = map
        df[self.column] = df[self.column].map(map)
        if join:
            if not np.array_equal(df[self.column].values, list(map.values())):
                df = pd.DataFrame(map.values(), columns = [self.column])\
                .merge(df, on = self.column, how = join)
        return df.sort_values(self.column)
            
class User(Object):
    def __init__(self, df, column):
        super().__init__(df, column)
        
class Item(Object):
    def __init__(self, df, column):
        super().__init__(df, column)
        
class Rating:
    def __init__(self, df, user, item, rating_column):
        self.df = df
        self.user = user
        self.item = item
        self.rating_column = rating_column
        self.num_users = df[user.column].nunique()
        self.num_items = df[item.column].nunique()
        
    def _reindex(self, df, column):
        df = df.copy()
        df[column] = df[column].astype("category").cat.codes
        return df
    
    @property
    def reindex(self):
        return self._reindex(
            self._reindex(self.df, self.user.column), 
            self.item.column
        )
    
    @property
    def user_map(self):
        return {v: i for i, v in
        enumerate(self.df[self.user.column].astype("category").cat.categories)}
        
    @property
    def item_map(self):
        return {v: i for i, v in
        enumerate(self.df[self.item.column].astype("category").cat.categories)}
    
    def _select_related(self, obj, column):
        merge = self.df.merge(obj, on = column, how = "left")
        return merge[obj.columns]
    
    @property
    def user_select_related(self):
        return self._select_related(self.user.df, self.user.column)
    
    @property
    def item_select_related(self):
        return self._select_related(self.item.df, self.item.column)