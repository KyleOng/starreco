# star-reco 
**star-reco** is a master thesis work + Python implementation for a series of deep learning rating-based recommendation system. **star-reco** is built on top of Pytorch and Pytorch-lightning for reproducing and developing various SOTA rating-based recommendation algorithms. 

Background
---
Star classification is a type of ratings scale commonly used by reviewers to rate certain items such as films, TV shows, restaurants, and hotels. Businesses are able to utilize user-item star ratings to build rating-based recommendation system, as these ratings can directly or indirectly reflect customers' preferences. This recommendation technique is also known as the **Collaborative Filtering (CF)** technique which recommendations are generated based on user-item interaction analysis. Correspondingly, various techniques for recommendation generation has been proposed throughout the years, in order to solve previous existing problems (For example, cold start and data sparsity). Hence, the purpose of this research is to understand, recreate and evaluate various SOTA recommendation algorithms in a coherent and unified repository.

Research Models
---
Current supported algorithms/models are:


|Research models|Linear|MLP|AE|CNN|
|-|:-:|:-:|:-:|:-:|
|Matrix Factorization (MF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/mf.py)> <[paper](https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf)>|:heavy_check_mark:||||
|Factorization Machine (FM) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/fm.py)> <[paper](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)>|:heavy_check_mark:||||
|Neural Collaborative Filtering (NCF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/ncf.py)> <[paper](https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf)>||:heavy_check_mark:|||
|Neural Matrix Factorization (NMF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/nmf.py)> <[paper](https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf)>|:heavy_check_mark:|:heavy_check_mark:|||
|Outer Product-based Neural Collaborative Filtering (ONCF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/oncf.py)> <[paper](https://arxiv.org/pdf/1808.03912.pdf)>||||:heavy_check_mark:|
|ConvolutionalNeural Networks based Deep Collaborative Filtering model (CNN-DCF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/cnndcf.py)> <[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9086604)>||||:heavy_check_mark:|
|Neural Factorization Machine (NFM) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/nfm.py)> <[paper](https://arxiv.org/pdf/1708.05027&ie=utf-8&sc_us=6917339300733978278.pdf)>|:heavy_check_mark:|:heavy_check_mark:|||
|Deep Factorization Machine (DeepFM) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/dfm.py)> <[paper](https://arxiv.org/pdf/1703.04247.pdf)>|:heavy_check_mark:|:heavy_check_mark:|||
|Extreme Deep Factorization Machine (xDeepFM) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/xdfm.py)> <[paper](https://arxiv.org/pdf/1803.05170.pdf)>|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|
|Wide & Deep<[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/wnd.py)> <[paper](https://arxiv.org/pdf/1606.07792.pdf%29/)>||:heavy_check_mark:|||





 
