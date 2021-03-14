# star-reco 
**star-reco** is a master thesis work + Python implementation for a series of deep learning rating-based recommendation system. **star-reco** is built on top of Pytorch and Pytorch-lightning for reproducing and developing various SOTA rating-based recommendation algorithms. 

Background
---

<img src="asset/19197320.jpg" width = "700"/>

Star classification is a type of ratings scale commonly used by reviewers to rate certain items such as films, TV shows, restaurants, and hotels. Businesses are able to utilize user-item star ratings to build rating-based recommendation system, as these ratings can directly or indirectly reflect customers' preferences. This recommendation technique is also known as the **Collaborative Filtering (CF)** technique which recommendations are generated based on user-item interaction analysis. Correspondingly, various techniques for recommendation generation has been proposed throughout the years and commercially deployed in real-life environment. Hence, the purpose of this research is to recreate and evaluation various SOTA recommendation algorithms in a coherent and unified repository.

Research Models
---
Current supported algorithms/models are:

|Research models|Linear|MLP|AE|CNN|
|-|:-:|:-:|:-:|:-:|
|Matrix Factorization (MF)|:heavy_check_mark:||||
|Factorization Machine (FM)|:heavy_check_mark:||||
|Neural Collaborative Filtering (NCF)||:heavy_check_mark:|||
|Wide & Deep||:heavy_check_mark:|||
|AutoRec|||:heavy_check_mark:||
|Deep AutoRec|||:heavy_check_mark:||
|Hybrid AutoRec|||:heavy_check_mark:||
|Variational Autoencoders|||:heavy_check_mark:||
|Convolutional Neural Collaborative Filtering (ConvNCF)||||:heavy_check_mark:|
|Double Embeddings Convolutional Neural Collaborative Filtering (DE ConvNCF)||||:heavy_check_mark:|
|Neural Matrix Factorization (NMF)|:heavy_check_mark:|:heavy_check_mark:|||
|Neural Factorization Machine (NFM)|:heavy_check_mark:|:heavy_check_mark:|||
|Deep Factorization Machine (DeepFM)|:heavy_check_mark:|:heavy_check_mark:|||
|Extreme Deep Factorization Machine (xDeepFM)|:heavy_check_mark:|:heavy_check_mark:|||
|Deep Collaborative Filtering via Marginalized Denoising Auto-encoder|:heavy_check_mark:||:heavy_check_mark:||
|Additional Stacked Denoising Autoencoders (aSDAE)|:heavy_check_mark:||:heavy_check_mark:||
|Convolutional Matrix Factorization (Conv MF)|:heavy_check_mark:|||:heavy_check_mark:|
|Hybrid-based Recommendation System based on AE (DHA-RS)||:heavy_check_mark:|:heavy_check_mark:||
|CNN Deep Collaborative Filtering (CNN-DCF)||:heavy_check_mark:||:heavy_check_mark:|
|Stacked Convolutional AE (SCAE)|||:heavy_check_mark:|:heavy_check_mark:|
|Feature Generated CNN (FGCNN)|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|
|Deep Hybrid Collaborative Filtering (Deep HCF)|:heavy_check_mark:|:heavy_check_mark:||:heavy_check_mark:|
|SDAE DE ConvMF||:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|


 
