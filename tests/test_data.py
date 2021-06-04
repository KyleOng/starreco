import pdb
import sys
sys.path.insert(0,"..")

from starreco.data import *

dm = StarDataModule("ml-1m", 
                    add_features = True, 
                    user_features_ignore = ["zipCode"],
                    item_features_ignore = [])
dm.setup()
