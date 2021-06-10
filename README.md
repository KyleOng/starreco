# starreco
![Python](https://img.shields.io/badge/python-3.6|3.7|3.8-blue?&logo=python)
![Pytorch](https://img.shields.io/badge/made%20with-Pytorch-critical?logo=pytorch)
![Lightning](https://img.shields.io/badge/made%20with-Pytorch%20Lightning-blueviolet?)
[![Version](https://img.shields.io/badge/version-v1.0.0-orange)](https://github.com/KyleOng/starreco) 
![GitHub repo size](https://img.shields.io/github/repo-size/KyleOng/starreco) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**starreco** stands for **S**tate-of-**T**he-**A**rt **R**eview **Reco**mmendation System.

**starreco** is a Pytorch lightning implementation for a series of SOTA deep learning rating-based recommendation systems. This repository also serves as a part of the author's master thesis work's literature review. 

Features
---
+ Up to 20+ recommendation models across 15 publications.
+ Built on top of [Pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning).
+ GPU acceleration execution.
+ Reducing memory usage for large sparse matrices.
+ Simple and understandable code.
+ Easy extension and code reusability.

Click [here](#start) to get started!

Research Models
---
|Research model |Description|Reference|
|-|-|-|
|[MF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/mf.py)|Matrix Factorization|<a href="#1">[1]</a>|
|[GMF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/gmf.py)|Generalized Matrix Factorization|<a href="#2">[2]</a>|
|[MLP](#https://github.com/KyleOng/starreco/blob/master/starreco/model/ncf.py)|Multilayer Perceptrons|<a href="#2">[2]</a>|
|[NeuMF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/nmf.py)|Neural Matrix Factorization|<a href="#2">[2]</a>|
|[FM](#https://github.com/KyleOng/starreco/blob/master/starreco/model/fm.py)|Factorization Machine|<a href="#3">[3]</a>|
|[NeuFM](#https://github.com/KyleOng/starreco/blob/master/starreco/model/nfm.py)|Neural Factorization Machine|<a href="#4">[4]</a>|
|[WDL](#https://github.com/KyleOng/starreco/blob/master/starreco/model/wdl.py)|Wide & Deep Learning|<a href="#5">[5]</a>|
|[DeepFM](#https://github.com/KyleOng/starreco/blob/master/starreco/model/dfm.py)|Deep Factorization Machine|<a href="#6">[6]</a>|
|[xDeepFM](#https://github.com/KyleOng/starreco/blob/master/starreco/model/xdfm.py)|Extreme Deep Factorization Machine|<a href="#7">[7]</a>|
|[FGCNN](#https://github.com/KyleOng/starreco/blob/master/starreco/model/fgcnn.py)|Feature Generation by using Convolutional Neural Network|<a href="#8">[8]</a>|
|[ONCF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/oncf.py)|Outer-based Product Neural Collaborative Filtering|<a href="#9">[9]</a>|
|[CNNDCF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/cnndcf.py)|Convolutional Neural Network based Deep Colloborative Filtering|<a href="#10">[10]</a>|
|[ConvMF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/cmf.py)|Convolutional Matrix Factorization|<a href="#11">[11]|
|[AutoRec](#https://github.com/KyleOng/starreco/blob/master/starreco/model/autorec.py)|AutoRec|<a href="#12">[12]|
|[DeepRec](#https://github.com/KyleOng/starreco/blob/master/starreco/model/deeprec.py)|DeepRec|<a href="#13">[13]|
|[CFN](#https://github.com/KyleOng/starreco/blob/master/starreco/model/cfn.py)|Collaborative Filtering Network|<a href="#14">[14]|
|[CDAE](#https://github.com/KyleOng/starreco/blob/master/starreco/model/cdae.py)|Collaborative Denoising AutoEncoder|<a href="#15">[15]|
|[CCAE](#https://github.com/KyleOng/starreco/blob/master/starreco/model/ccae.py)|Collaborative Convolutional AutoEncoder|<a href="#16">[16]|
|[SDAECF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/sdaecf.py)|Stacked Denoising AutoEncoder for Collaborative Filtering|<a href="#17">[17]|
|[mDACF](#https://github.com/KyleOng/starreco/blob/master/starreco/model/mdacf.py)|marginalized Denoising AutoEncoder Collaborative Filtering|<a href="#18">[18]|
|[GMF++](#https://github.com/KyleOng/starreco/blob/master/starreco/model/gmfpp.py)|Generalized Matrix Factorization ++|<a href="#1">[19]|
|[MLP++](#https://github.com/KyleOng/starreco/blob/master/starreco/model/ncfpp.py)|Multilayer Perceptrons ++|<a href="#1">[19]|
|[NeuMF++](#https://github.com/KyleOng/starreco/blob/master/starreco/model/nmfpp.py)|Neural Matrix Factorization ++|<a href="#20">[20]|

Architecture
---
<p align="center">
  <img src="asset/architecture.png" alt="starrreco v0.1 architecture">
  <br>
  <b>Figure</b>: starreco overall architecture
</p>

Datasets
---
- **Movielen Dataset**: A movie rating dataset collected from the Movielens websites by the GroupLensResearch Project  at University of Minnesota. The datasets were collected over various time periods, depending on the sizes given. **Movielen 1M Dataset**** has been chosen. It contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

- **Amazon Dataset**: Amazon dataset is the consumption records from Amazon.com, which contains product reviews and metadata. It includes 142.8 million reviewscollected from May 1996 to July 2004. The Amazon dataset is also categorized into smaller datasets with different categories of productsuch as books, electronics, movie, etc. Hence, researchers can select smaller datasetsbased on their interest of domain of research. **Amazon Instant Video**, **Amazon Android Apps** and **Amazon Digital Music** have been chosen.

Getting Started<a name="start"></a>
---
### Installation 

Create virtual environment

```bash
python3 -m virtualenv env # Python 3.6 and above
```

Activate virtual environment
```bash
source env/bin/activate # Linux
./env/Scripts/activate # Windows
```

Clone and install necessary python packages
```bash
git clone https://github.com/KyleOng/star-reco
pip install -r requirements.txt
```

### Example
```python

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from starreco.model import *
from starreco.data import *
    
# data module
data_module = DataModule("ml-1m")
data_module.setup()
    
# model
model = MF([data_module.dataset.rating.num_users, 
            data_module.dataset.rating.num_items])

# setup
# checkpoint callback
checkpoint_callback = ModelCheckpoint(dirpath = f"checkpoints/mf/version_{i + current_version}",
                                      monitor = "val_loss_",
                                      filename = "mf-{epoch:02d}-{train_loss_:.4f}-{val_loss_:.4f}")
# logger
logger = TensorBoardLogger("training_logs", 
                           name = "mf", 
                           log_graph = True)
# trainer
trainer = Trainer(logger = logger,
                  gpus = -1 if torch.cuda.is_available() else None, 
                  max_epochs = 100, 
                  progress_bar_refresh_rate = 2,
                  callbacks=[checkpoint_callback])
trainer.fit(module, data_module)

# evaluate
module_test = MF.load_from_checkpoint(checkpoint_callback.best_model_path)
trainer.test(module_test, datamodule = data_module)
```

References
---
<a id="1">[1]</a> Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37. 

<a id="2">[2]</a> He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).

<a id="3">[3]</a> Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.

<a id="4">[4]</a> He, X., & Chua, T. S. (2017, August). Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval (pp. 355-364).

<a id="5">[5]</a> Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Shah, H. (2016, September). Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems (pp. 7-10).

<a id="6">[6]</a> Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247.

<a id="7">[7]</a> Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018, July). xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1754-1763).

<a id="8">[8]</a> Liu, B., Tang, R., Chen, Y., Yu, J., Guo, H., & Zhang, Y. (2019, May). Feature generation by convolutional neural network for click-through rate prediction. In The World Wide Web Conference (pp. 1119-1129).

<a id="9">[9]</a> He, X., Du, X., Wang, X., Tian, F., Tang, J., & Chua, T. S. (2018). Outer product-based neural collaborative filtering. arXiv preprint arXiv:1808.03912.

<a id="10">[10]</a> Wu, Y., Wei, J., Yin, J., Liu, X., & Zhang, J. (2020). Deep Collaborative Filtering Based on Outer Product. IEEE Access, 8, 85567-85574.

<a id="11">[11]</a> Kim, D., Park, C., Oh, J., Lee, S., & Yu, H. (2016, September). Convolutional matrix factorization for document context-aware recommendation. In Proceedings of the 10th ACM conference on recommender systems (pp. 233-240).

<a id="12">[12]</a> Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May). Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th international conference on World Wide Web (pp. 111-112).

<a id="13">[13]</a> Kuchaiev, O., & Ginsburg, B. (2017). Training deep autoencoders for collaborative filtering. arXiv preprint arXiv:1708.01715.

<a id="14">[14]</a> Strub, F., Mary, J., & Gaudel, R. (2016). Hybrid collaborative filtering with autoencoders. arXiv preprint arXiv:1603.00806.

<a id="15">[15]</a> Wu, Yao, et al. "Collaborative denoising auto-encoders for top-n recommender systems." Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.

<a id="16">[16]</a> Zhang, S. Z., Li, P. H., & Chen, X. N. (2019, December). Collaborative Convolution AutoEncoder for Recommendation Systems. In Proceedings of the 2019 8th International Conference on Networks, Communication and Computing (pp. 202-207).

<a id="16">[17]</a> Strub, F., & Mary, J. (2015, December). Collaborative filtering with stacked denoising autoencoders and sparse inputs. In NIPS workshop on machine learning for eCommerce.

<a id="18">[18]</a> Li, S., Kawale, J., & Fu, Y. (2015, October). Deep collaborative filtering via marginalized denoising auto-encoder. In Proceedings of the 24th ACM international on conference on information and knowledge management (pp. 811-820).

<a id="19">[19]</a> Liu, Y., Wang, S., Khan, M. S., & He, J. (2018). A novel deep hybrid recommender system based on auto-encoder with neural collaborative filtering. Big Data Mining and Analytics, 1(3), 211-221.

<a id="20">[20]</a> To be published.

Github References
---
- https://github.com/khanhnamle1994/MetaRec
- https://github.com/shenweichen/DeepCTR-Torch
- https://github.com/makgyver/rectorch
- https://github.com/rixwew/pytorch-fm
- https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation
- https://github.com/shenweichen/DeepCTR/
- https://github.com/cartopy/ConvMF/