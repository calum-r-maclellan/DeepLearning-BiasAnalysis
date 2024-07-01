# Overview
Foundation models (FMs) serve an important role in data analytics for healthcare. 
Typically built upon deep learning algorithms, FMs help to reduce the burden of extensive data and compute resources required for deep learning by 

Both are promising attributes for healthcare AI. 
However, care must be taken to ensure that the dataset used in development does not pose the risk of causing dataset shift. This
In fact, it was recently shown that a popular FM (Google Health: https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md) encoded such biases in its predictions [1]. 
The authors demonstrated that the predictions favoured particular races and genders.

It is important for us to understand whether this problem persists in other datasets, as this would bring into question the reliability of the FM in the wild. 
To this end, we study whether the findings in [1] translate into another open-source radiography dataset, namely the MIMIC-CXR database [2].

# Methods
We follow the general framework described in [2] but using the MIMIC-CXR dataset instead of CheXpert. The workflow is as follows:
- train a DenseNet as in conventional setup: this serves as our baseline by which to compare with the foundaton model.
- extract embeddings from DenseNet: we will use these in our feature space analysis later to discover if the key principal directions in feature space correlate with our demographics, hinting at bias
- extract embeddings from the FM (image embeddings stored in the high-level features): same as above.
- train an MLP on the FM embeddings: we will use the resulting model to measure performance on resampled test set.

Given the trained models, we then perform resampling of the test set to ensure equal demographic and disease prevalence amongst patient subgroups.

To examine the effects of potential biasing toward these subgroups
- feature space interrogation:
- prediction accuracy across subgroups.

# Results
## Features 
Here we compare the PCA transformed embeddings between the conventional and foundation model. 

### PCA
![](https://github.com/calum-r-maclellan/FM-bias-analysis-MIMIC-CXR/blob/main/figs/densenet-pca/pca-1-densenet-disease-marginal.pdf)




### t-SNE


## Classification results
![](https://github.com/calum-r-maclellan/SGAN-COVID19/blob/main/pics/class_perf.png)

## Performance 
![](https://github.com/calum-r-maclellan/SGAN-COVID19/blob/main/pics/gradcam++.png)

# Conclusions
- despite seeing significantly less non-COVID labelled data, SGAN demonstrates strong similarity in diagnostic accuracy to the supervised equivalent.
- SGAN presents as a highly promising architecture for performing semi-supervised learning on this task.
- however it needs further investigation due to the potential fitting of confounding variables. 
- this work lays the foundations to build upon this model and devise novel methods to improve the classification performance, and enhance the ability to detect underlying covariates most strongly linked to COVID-19.
