# Abstract

In this project, we describe the concept of a bias analysis within the context of foundation models (FMs) for chest radiography. 
We begin by introducing what FMs are and how they can manifest so-called biases, before discussing why it is in our best interest 
to mitigate them. 
We then describe what a bias analysis entails, and demonstrate its utility through a full worked example using publicly available resources. 
We show that the studied foundation model from Google Health encodes no biases on the MIMIC-CXR dataset: if considered for real-world use, 
the resulting evidence would support the safe implementation of this model into a clinical setting with similar patient populations. 
On this basis, we hope to convince AI vendors of the expected utility of involving such evidence in the development (and deployment) phase of clinically targeted FMs for optimising their integration into practice.

More details of the background and methodology are presented in the accompanying pdf file.

# Results 
We present the main results to the bias analysis in Figures 2 (PCA visualisations) and 3 (generalisation performance).
For the PCA results, we are looking to demonstrate two key findings: (1) significant overlap- ping across our patient subgroups; and (2) significant differences across our disease labels. In the former, this demonstrates that the FM does not leverage sensitive attributes to dis- criminate between each pathology label; in the latter, non-overlapping support across the distributions suggests the model has learned to associate unique features to each pathology. For the performance analysis, our goal is to show no disparities across the patient subgroups, meaning we observe consistent performance no matter the demographic attributes of the patients.

![Comparison](figs/densenet-pca/pca-1+2-densenet-age.png)
