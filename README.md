# star-reco 
**star-reco** is a master thesis work + Python implementation for a series of deep learning rating-based recommendation system. **star-reco** is built on top of Pytorch and Pytorch-lightning for reproducing and developing various SOTA rating-based recommendation algorithms. 

This work also acts as a part of the master thesis work's literature review.

Background
---
Star classification is a type of ratings scale commonly used by reviewers to rate certain items such as films, TV shows, restaurants, and hotels. Businesses are able to utilize user-item star ratings to build rating-based recommendation system, as these ratings can directly or indirectly reflect customers' preferences. This recommendation technique is also known as the **Collaborative Filtering (CF)** technique which recommendations are generated based on user-item interaction analysis. Correspondingly, various techniques for recommendation generation has been proposed throughout the years, in order to solve previous existing problems (For example, cold start and data sparsity). Hence, the purpose of this research is to understand, recreate and evaluate various SOTA recommendation algorithms in a coherent and unified repository.

Research Models
---
Current supported algorithms/models are:

<sup> * asterisk symbol indicates code modification for originally CTR models to Rating Prediction Models</sup>
|Research models|Linear|MLP|AE|CNN|
|-|:-:|:-:|:-:|:-:|
|Matrix Factorization (MF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/mf.py)> <[paper](https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf)>|:heavy_check_mark:||||
|Factorization Machine (FM) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/fm.py)> <[paper](https://sdcast.ksdaemon.ru/wp-content/uploads/2020/02/Rendle2010FM.pdf)>|:heavy_check_mark:||||
|Neural Collaborative Filtering (NCF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/ncf.py)> <[paper](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)>||:heavy_check_mark:|||
|Wide & Deep * <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/wnd.py)> <[paper](https://arxiv.org/pdf/1606.07792.pdf%29/)>||:heavy_check_mark:|||
|Outer Product-based Neural Collaborative Filtering (ONCF) * <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/oncf.py)> <[paper](https://arxiv.org/pdf/1808.03912.pdf)>||||:heavy_check_mark:|
|ConvolutionalNeural Networks based Deep Collaborative Filtering model (CNN-DCF) * <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/cnndcf.py)> <[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9086604)>||||:heavy_check_mark:|
|Neural Matrix Factorization (NMF) <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/nmf.py)> <[paper](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)>|:heavy_check_mark:|:heavy_check_mark:|||
|Neural Factorization Machine (NFM) * <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/nfm.py)> <[paper](https://arxiv.org/pdf/1708.05027&ie=utf-8&sc_us=6917339300733978278.pdf)>|:heavy_check_mark:|:heavy_check_mark:|||
|Deep Factorization Machine (DeepFM) * <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/dfm.py)> <[paper](https://arxiv.org/pdf/1703.04247.pdf)>|:heavy_check_mark:|:heavy_check_mark:|||
|Extreme Deep Factorization Machine (xDeepFM) * <[code](https://github.com/KyleOng/star-reco/blob/master/starreco/model/xdfm.py)> <[paper](https://arxiv.org/pdf/1803.05170.pdf)>|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|

Datasets
---
- **Movielen Dataset**: A movie rating dataset collected from the Movielens websites by the GroupLensResearch Project  at University of Minnesota. The datasets were collected over various time periods, depending on the sizes given. Movielen 1M Dataset is chosen for evaluation. It contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

- **Epinions Dataset**: A product review dataset collected by Paolo Massa in 5 weeks (November/December 2003) from Epinions.com website.Epinions dataset is used in many recommendation researches works due toits trust factor for better recommendation accuracy. The dataset contains two sources of data which are trust data and rating data. In the trustdataset, all trust values are stored and distrust values are discarded. As for the rating data, each item rating is rated in the scale of 1 to 5. Epinions Dataset also contains 49,290 users who rated a total of 139,738 different items at least once, writing 664,824 reviews and 487,181 issued trust statements.

- **Amazon Dataset**: Amazon dataset is the consumption records from Amazon.com, which contains product reviews and metadata. Itincludes142.8 million reviewscollected from May 1996 to July 2004. The Amazon dataset is also categorized into smaller datasets with different categories of productsuch as books, electronics, movie, etc. Hence, researchers can select smaller datasetsbased on their interest of domain of research. The 3 subset of this Dataset chosen for evaluation are **Amazon Instant Video**, **Amazon Android Apps** and **Amazon Digital Music**.

- **Book Crossing Dataset**.

Acknowledgements
---
This work is inspired by the following links for creating a unified and comprehensive repository for rating prediction recommendation.
- https://github.com/RUCAIBox/RecBole
- https://github.com/khanhnamle1994/MetaRec
- https://github.com/shenweichen/DeepCTR-Torch

Special thanks to the following repositories for github code references and model/algorithms understanding.
- https://github.com/khanhnamle1994/MetaRec
- https://github.com/rixwew/pytorch-fm
- https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation
- https://github.com/shenweichen/DeepCTR/blob/master/deepctr/models/fgcnn.py






 
