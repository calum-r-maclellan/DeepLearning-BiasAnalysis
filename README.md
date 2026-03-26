# Introduction

Deep learning has revolutionised the field of medical imaging analysis due to its ability to extract meaningful predictive patterns automatically from large clinical databases. In this setting, the goal is to learn a model for transforming a medical image (e.g., chest X-ray) into a diagnostic label indicating the presence or absence of pathology. Given the learned model, one can then infer labels on new patient data, creating opportunities for real-time diagnostic use in the clinical setting. 
However, deep learning models (by design) tend to suffer from over-parameterisation, where the size of the model parameter space outmatches the size of the input feature space. As a result, they tend to leverage noise in the data in order to separate categories of interest, such as disease states (healthy or not). 
This can often cause the model to learn patterns which (though statistically correct) have no relevance to the predictive task at hand. 

To make this concept clearer, consider the diagram below. 
This describes a deep learning model that has been trained to separate healthy patients from those with Cardiomegaly, based on their CXR data. Although it has achieved a certain degree of predictive accuracy at doing this, the model has also learned that the presence of Cardiomegaly is largely attributed to the patient's race, where more White patients have Cardiomegaly than Black patients. 
This is a spurious signal, and suggests that the training dataset is biased towards White patients: the model has picked this statistical signal up, and propagates it into all future predictions. 


In fact, it was recently shown by a group of researchers from the UK [4] that a popular open-source foundation model developed by Google Health [9] has a propensity to make biased predictions on a dataset outside of its training set, called CheXpert. 
A bias analysis helps identify such signals, and to inform data scientists/ML engineers about where they ought to better prepare the dataset for retraining.
![Disease-pca-model-comparison](figs/fig1.biasAnalysis.png)

In this project, we ask the question: 'is the model biased toward any dataset outside of its support?' 
We follow the bias analysis technique in [4] using another dataset, called MIMIC-CXR, and study the foundation (pre-trained) model's corresponding predictions against a baseline model trained from scratch on MIMIC-CXR. 

# Results 
We present the main results to the bias analysis in Figures 2 (PCA visualisations) and 3 (generalisation performance).
For the PCA results, we are looking to demonstrate two key findings: (1) significant overlap- ping across our patient subgroups; and (2) significant differences across our disease labels. In the former, this demonstrates that the FM does not leverage sensitive attributes to dis- criminate between each pathology label; in the latter, non-overlapping support across the distributions suggests the model has learned to associate unique features to each pathology. For the performance analysis, our goal is to show no disparities across the patient subgroups, meaning we observe consistent performance no matter the demographic attributes of the patients.

![Disease-pca-model-comparison](figs/fig2.results.png)

![Disease-pca-model-comparison](figs/fig3.results.png)


# Conclusions
The purpose of this project was to examine whether an open-source foundation model (FM) yielded preferential predictions towards certain demographics of patients in an out-of-distribution dataset. We follow-on from previous work [4], who revealed that the FM provided by Google Health [9] did indeed exhibit biased predictions on the so-called CheXpert dataset. Our  goal was to study the extent of such biases within a separate validation dataset called MIMIC-CXR [10].

Through statistical signifance testing, we demonstrated that the evidence strongly suggested that the FM encoded no biases on this particular dataset, helping support potential decisions to implement this FM in settings with similar patient populations. 

# References 
[1] C ̧allı E, Sogancioglu E, van Ginneken B, van Leeuwen KG, and Murphy K. Deep learning for chest x-ray analysis: a survey. Med. Image Anal., 72(102125):259–265, 2021.

[2] M. Moor, O. Banerjee, Z.S.H. Abad, and et al. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265, 2023.

[3] D. C. Castro, I. Walker, and B. Glocker. Causality matters in medical imaging. Nature Communications, 11(3673), 2020.

[4] B. Glocker, C. Jones, M. Roschewitz, and S. Winzeck. Risk of bias in chest radiography deep learning foundation models. Radiology: Artificial Intelligence, 5(6), 2023.

[5] Y. Zong, Y. Yang, and T. Hospedales. Medfair: Benchmarking fairness for medical imaging. ICLR, 2023.

[6] R. Dutt, O. Bohdal, S. A. Tsaftaris, and T. Hospedalas. Fairtune: optimizing parameter-efficient fine-tuning for fairness in medical imaging analysis. ICLR, 2024.

[7] W.F. Wiggins and A.S. Tejani. On the opportunities and risks of foundation models for natural language processing in radiology. Radiology: Artificial Intelligence, 4(4), 2022.

[8] C. J. Soelistyo, G. Vallardi, G. Charras, and A.R. Lowe. Learning biophysical determi- nants of cell fate with deep neural networks. Nature Machine Intelligence, 4:636–644, 2022.

[9] A.B. Sellergren and et al. Simplified transfer learning for chest radiography models using less data. Radiology, 2022.

[10] Johnson A.E., Pollard T.J., and Lungren M.P. Deng C.Y. Mark R.G. Horng S. Berkowitz S., Greenbaum N.R. Mimic-cxr: A large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042, 2019.

[11] G. Huang and et al. Densely connected convolutional networks. CVPR, 2017.
