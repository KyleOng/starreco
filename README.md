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
    <th rowspan = 3>Research models</th>
    <th rowspan = 3>Direct links</th>
    <th colspan = 4>Hybrid models</th>
    <th rowspan = 3>Reference</th>
  </tr>
  <tr>
    <th rowspan = 2>Linear</th>
    <th colspan = 3>Non-linear</th>
  </tr>
  <tr>
    <th>SLP/MLP</th>
    <th>AE</th>
    <th>CNN</th>
  </tr>
  <tr>
    <td>MF</td>
    <td>
      <a href="https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mf.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td></td>
    <td><a href="#1">[1]</a></td>
  </tr>
  <tr>
    <td>FM</td>
    <td>
      <a href="https://sdcast.ksdaemon.ru/wp-content/uploads/2020/02/Rendle2010FM.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/fm.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td></td>
    <td>[2]</td>
  </tr>
  <tr>
    <td>GMF</td>
    <td>
      <a href="https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/gmf.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>[3]</td>
  <tr>
    <td>NCF/MLP</td>
    <td>
      <a href="https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/ncf.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>[3]</td>
  </tr>
  <tr>
    <td>WDL</td>
    <td>
      <a href="https://arxiv.org/pdf/1606.07792.pdf%29/">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/wdl.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>[4]</td>
  </tr>
  <tr>
    <td>AutoRec</td>
    <td>
      <a href="http://users.cecs.anu.edu.au/~u5098633/papers/www15.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/ae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[5]</td>
  </tr>
  <tr>
    <td>DeepRec</td>
    <td>
      <a href="https://arxiv.org/pdf/1708.01715.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/dae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[6]</td>
  </tr>
  <tr>
    <td>CFN</td>
    <td>
      <a href="https://arxiv.org/pdf/1603.00806.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/hae.py">code</a> 
    </td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[7]</td>
  </tr>
  <tr>
    <td>Mult-VAE</td>
    <td>
      <a href="https://arxiv.org/pdf/1802.05814.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mvae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[8]</td>
  </tr>
  <tr>
    <td>Mult-DAE</td>
    <td>
      <a href="https://arxiv.org/pdf/1802.05814.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mdae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[8]</td>
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
    <td>:heavy_check_mark:</td>
    <td>[9]</td>
  </tr>
  <tr>
    <td>NeuMF</td>
    <td>
      <a href="https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/nmf.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>[3]</td>
  </tr>
  <tr>
    <td>NeuFM</td>
    <td>
      <a href="https://arxiv.org/pdf/1708.05027.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/nfm.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>[10]</td>
  </tr>
  <tr>
    <td>DeepFM</td>
    <td>
      <a href="https://arxiv.org/pdf/1703.04247.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/dfm.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>[11]</td>
  </tr>
  <tr>
    <td>mDA-CF</td>
    <td>
      <a href="https://dl.acm.org/doi/pdf/10.1145/2806416.2806527">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/fgcnn.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[12]</td>
  </tr>
  
  <tr>
    <td>aSDAE+MF</td>
    <td>
      <a href="https://ojs.aaai.org/index.php/AAAI/article/view/10747/10606">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/asdaemf.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[13]</td>
  </tr>
  <tr>
    <td>ConvMF</td>
    <td>
      <a href="http://uclab.khu.ac.kr/resources/publication/C_351.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/cmf.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>[14]</td>
  </tr>
  <tr>
    <td>DE-ConvMF</td>
    <td>
      <a href="https://download.atlantis-press.com/article/125910161.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/decmf.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>[15]</td>
  </tr><tr>
    <td>GMF++</td>
    <td>
      <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8361573">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/cmfpp.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[16]</td>
  </tr>
  <tr>
    <td>MLP++</td>
    <td>
      <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8361573">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/mlppp.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>[16]</td>
  </tr>
  <tr>
    <td>CNN-DCF</td>
    <td>
      <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9086604">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/cnndcf.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>[17]</td>
  </tr>
  <tr>
    <td>xDeepFM</td>
    <td>
      <a href="https://arxiv.org/pdf/1803.05170.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/xdfm.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>[18]</td>
  </tr>
  <tr>
    <td>FGCNN</td>
    <td>
      <a href="https://arxiv.org/pdf/1904.04447.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/fgcnn.py">code</a>
    </td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>[19]</td>
  </tr>
  <tr>
    <td>CCAE</td>
    <td>
      <a href="">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/ccae.py">code</a>
    </td>
    <td></td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td>[20]</td>
  </tr>
  <tr>
    <td>SDAE-DE-ConvMF</td>
    <td>
      <a href="https://download.atlantis-press.com/article/125910161.pdf">paper</a> | 
      <a href="https://github.com/KyleOng/star-reco/blob/master/starreco/model/sdaedecmf.py">code</a>
    </td>
    <td>:heavy_check_mark:</td>
    <td></td>
    <td>:heavy_check_mark:</td>
    <td>:heavy_check_mark:</td>
    <td>[15]</td>
  </tr>
</table>

Datasets
---
- **Movielen Dataset**: A movie rating dataset collected from the Movielens websites by the GroupLensResearch Project  at University of Minnesota. The datasets were collected over various time periods, depending on the sizes given. Movielen 1M Dataset is chosen for evaluation. It contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

- **Epinions Dataset**: A product review dataset collected by Paolo Massa in 5 weeks (November/December 2003) from Epinions.com website.Epinions dataset is used in many recommendation researches works due toits trust factor for better recommendation accuracy. The dataset contains two sources of data which are trust data and rating data. In the trustdataset, all trust values are stored and distrust values are discarded. As for the rating data, each item rating is rated in the scale of 1 to 5. Epinions Dataset also contains 49,290 users who rated a total of 139,738 different items at least once, writing 664,824 reviews and 487,181 issued trust statements.

- **Amazon Dataset**: Amazon dataset is the consumption records from Amazon.com, which contains product reviews and metadata. It includes 142.8 million reviewscollected from May 1996 to July 2004. The Amazon dataset is also categorized into smaller datasets with different categories of productsuch as books, electronics, movie, etc. Hence, researchers can select smaller datasetsbased on their interest of domain of research. The 3 subsets of this Dataset chosen for evaluation are **Amazon Instant Video**, **Amazon Android Apps** and **Amazon Digital Music**.

- **Book Crossing Dataset**.

Acknowledgements
---
This work is inspired by the following links.
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

References
---
[1] Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.

[2] Rendle, S. (2010, December). Factorization machines. In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.

[3] He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web (pp. 173-182).

[4] Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Shah, H. (2016, September). Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems (pp. 7-10).

[5] Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May). Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th international conference on World Wide Web (pp. 111-112).

[6] Kuchaiev, O., & Ginsburg, B. (2017). Training deep autoencoders for collaborative filtering. arXiv preprint arXiv:1708.01715.

[7] Strub, F., Mary, J., & Gaudel, R. (2016). Hybrid collaborative filtering with autoencoders. arXiv preprint arXiv:1603.00806.

[8] Liang, D., Krishnan, R. G., Hoffman, M. D., & Jebara, T. (2018, April). Variational autoencoders for collaborative filtering. In Proceedings of the 2018 world wide web conference (pp. 689-698).

[9] He, X., Du, X., Wang, X., Tian, F., Tang, J., & Chua, T. S. (2018). Outer product-based neural collaborative filtering. arXiv preprint arXiv:1808.03912.

[10] He, X., & Chua, T. S. (2017, August). Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval (pp. 355-364).

[11] Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247.

[12] Li, S., Kawale, J., & Fu, Y. (2015, October). Deep collaborative filtering via marginalized denoising auto-encoder. In Proceedings of the 24th ACM international on conference on information and knowledge management (pp. 811-820).

[13] Dong, X., Yu, L., Wu, Z., Sun, Y., Yuan, L., & Zhang, F. (2017, February). A hybrid collaborative filtering model with deep structure for recommender systems. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 31, No. 1).

[14] Kim, D., Park, C., Oh, J., Lee, S., & Yu, H. (2016, September). Convolutional matrix factorization for document context-aware recommendation. In Proceedings of the 10th ACM conference on recommender systems (pp. 233-240).

[15] Zhao, J., Liu, Z., Chen, H., Zhang, J., & Wen, Q. (2019, June). Hybrid recommendation algorithms based on ConvMF deep learning model. In 2019 International Conference on Wireless Communication, Network and Multimedia Engineering (WCNME 2019) (pp. 151-154). Atlantis Press.

[16] Liu, Y., Wang, S., Khan, M. S., & He, J. (2018). A novel deep hybrid recommender system based on auto-encoder with neural collaborative filtering. Big Data Mining and Analytics, 1(3), 211-221.

[17] Wu, Y., Wei, J., Yin, J., Liu, X., & Zhang, J. (2020). Deep Collaborative Filtering Based on Outer Product. IEEE Access, 8, 85567-85574.

[18] Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018, July). xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge 
Discovery & Data Mining (pp. 1754-1763).

[19] Liu, B., Tang, R., Chen, Y., Yu, J., Guo, H., & Zhang, Y. (2019, May). Feature generation by convolutional neural network for click-through rate prediction. In The World Wide Web Conference (pp. 1119-1129).

[20] Zhang, S. Z., Li, P. H., & Chen, X. N. (2019, December). Collaborative Convolution AutoEncoder for Recommendation Systems. In Proceedings of the 2019 8th International Conference on Networks, Communication and Computing (pp. 202-207).
