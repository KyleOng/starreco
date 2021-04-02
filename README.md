# star-reco :star: :robot:
A Pytorch and lightning implementation for a series of deep learning `star` or rating-based `reco`mmendation systems. This work also acts as a part of the master thesis work's literature review.

Background
---
Star classification is a type of ratings scale commonly used by reviewers to rate certain items such as films, TV shows, restaurants, and hotels. Businesses are able to utilize user-item star ratings to build rating-based recommendation system, as these ratings can directly or indirectly reflect customers' preferences. This recommendation technique is also known as the **Collaborative Filtering (CF)** technique which recommendations are generated based on user-item interaction analysis. Correspondingly, various techniques for recommendation generation has been proposed throughout the years, in order to solve previous existing problems (For example, cold start and data sparsity). Hence, the purpose of this research is to understand, recreate and evaluate various SOTA recommendation algorithms in a coherent and unified repository.

Research Models
---
Current supported research models:
<table>
  <tr>
    <th rowspan = 2>Research models</th>
    <th rowspan = 2>Details</th>
    <th colspan = 4>Models</th>
  </tr>
  <tr>
    <th>Linear</th>
    <th>Nonlinear/FC/MLP</th>
    <th>AE</th>
    <th>CNN</th>
  </tr>
  <tr>
    <td>MF</td>
    <td>
      <a href="https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>FM</td>
    <td>
      <a href="https://sdcast.ksdaemon.ru/wp-content/uploads/2020/02/Rendle2010FM.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/fm.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>GMF</td>
    <td>
      <a href="https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/gmf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  <tr>
    <td>NCF/MLP</td>
    <td>
      <a href="https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/ncf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>WDL</td>
    <td>
      <a href="https://arxiv.org/pdf/1606.07792.pdf%29/">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/wdl.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>AutoRec</td>
    <td>
      <a href="http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/ae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  
  <tr>
    <td>DeepRec</td>
    <td>
      <a href="https://arxiv.org/pdf/1708.01715.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/dae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>CFN</td>
    <td>
      <a href="https://arxiv.org/pdf/1603.00806.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/hae.py">code</a> 
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Mult-VAE</td>
    <td>
      <a href="https://arxiv.org/pdf/1802.05814.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mvae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Mult-DAE</td>
    <td>
      <a href="https://arxiv.org/pdf/1802.05814.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mdae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ONCF</td>
    <td>
      <a href="https://arxiv.org/pdf/1808.03912.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/oncf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>CNN-DCF</td>
    <td>
      <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9086604">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/cnndcf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>NeuMF</td>
    <td>
      <a href="https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/nmf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>DeepFM</td>
    <td>
      <a href="https://arxiv.org/pdf/1703.04247.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/dfm.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>ConvMF</td>
    <td>
      <a href="http://uclab.khu.ac.kr/resources/publication/C_351.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/cmf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>DE-ConvMF</td>
    <td>
      <a href="https://download.atlantis-press.com/article/125910161.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/decmf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>xDeepFM</td>
    <td>
      <a href="https://arxiv.org/pdf/1803.05170.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/xdfm.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>FGCNN</td>
    <td>
      <a href="https://arxiv.org/pdf/1904.04447.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/fgcnn.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>mDA-CF</td>
    <td>
      <a href="https://arxiv.org/pdf/1904.04447.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/fgcnn.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  
  <tr>
    <td>GMF++</td>
    <td>
      <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8361573">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/cmfpp.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>MLP++</td>
    <td>
      <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8361573">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mlppp.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>SDAE-DE-ConvMF</td>
    <td>
      <a href="https://download.atlantis-press.com/article/125910161.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/sdaedecmf.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>
<sup> * asterisk symbol indicates code modification for originally CTR models to Rating Prediction models</sup>
  
Datasets
---
- **Movielen Dataset**: A movie rating dataset collected from the Movielens websites by the GroupLensResearch Project  at University of Minnesota. The datasets were collected over various time periods, depending on the sizes given. Movielen 1M Dataset is chosen for evaluation. It contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

- **Epinions Dataset**: A product review dataset collected by Paolo Massa in 5 weeks (November/December 2003) from Epinions.com website.Epinions dataset is used in many recommendation researches works due toits trust factor for better recommendation accuracy. The dataset contains two sources of data which are trust data and rating data. In the trustdataset, all trust values are stored and distrust values are discarded. As for the rating data, each item rating is rated in the scale of 1 to 5. Epinions Dataset also contains 49,290 users who rated a total of 139,738 different items at least once, writing 664,824 reviews and 487,181 issued trust statements.

- **Amazon Dataset**: Amazon dataset is the consumption records from Amazon.com, which contains product reviews and metadata. It includes 142.8 million reviewscollected from May 1996 to July 2004. The Amazon dataset is also categorized into smaller datasets with different categories of productsuch as books, electronics, movie, etc. Hence, researchers can select smaller datasetsbased on their interest of domain of research. The 3 subset of this Dataset chosen for evaluation are **Amazon Instant Video**, **Amazon Android Apps** and **Amazon Digital Music**.

- **Book Crossing Dataset**.

Acknowledgements
---
This work is inspired by the following links for creating a unified and comprehensive repository for rating prediction recommendation.
- https://github.com/RUCAIBox/RecBole
- https://github.com/khanhnamle1994/MetaRec
- https://github.com/shenweichen/DeepCTR-Torch

Special thanks to the following repositories for github code references and model/algorithms understanding.
- https://github.com/khanhnamle1994/MetaRec
- https://github.com/makgyver/rectorch
- https://github.com/rixwew/pytorch-fm
- https://github.com/RaptorMai/Deep-AutoEncoder-Recommendation
- https://github.com/shenweichen/DeepCTR/
- https://github.com/cartopy/ConvMF/





 
