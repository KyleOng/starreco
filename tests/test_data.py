import pdb
import argparse
import sys
sys.path.insert(0,"..")

from starreco.data import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Testing data")
    parser.add_argument("--option", type = str, default = "ml-1m", help = "dataset option")
    parser.add_argument("--add_features", type = bool, default = False, help = "add features")
    parser.add_argument('--user_ignore', nargs="*", default = [], help = "list of user features to be ignore")
    parser.add_argument('--item_ignore', nargs="*", default = [], help = "list of user features to be ignore")

    args = parser.parse_args()

    dm = StarDataModule(args.option, 
                        add_features = args.add_features, 
                        user_features_ignore = args.user_ignore,
                        item_features_ignore = args.item_ignore)
    dm.setup()