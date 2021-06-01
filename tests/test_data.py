import pdb
import sys
sys.path.insert(0,"..")

from starreco.data import *

dm = StarDataModule("ml-1m")
dm.setup()