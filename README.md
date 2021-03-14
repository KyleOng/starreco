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
|Matrix Factorization (MF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/mf.py)><[paper](https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf)>|:heavy_check_mark:||||



 
