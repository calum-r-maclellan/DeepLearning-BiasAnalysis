# Abstract

In this project, we describe the concept of a bias analysis within the context of foundation models (FMs) for chest radiography. 
We begin by introducing what FMs are and how they can manifest so-called biases, before discussing why it is in our best interest 
to mitigate them. 
We then describe what a bias analysis entails, and demonstrate its utility through a full worked example using publicly available resources. 
We show that the studied foundation model from Google Health encodes no biases on the MIMIC-CXR dataset: if considered for real-world use, 
the resulting evidence would support the safe implementation of this model into a clinical setting with similar patient populations. 
On this basis, we hope to convince AI vendors of the expected utility of involving such evidence in the development (and deployment) phase of clinically targeted FMs for optimising their integration into practice.

# 1. Introduction

**Background on foundation models.** 
Deep learning (DL) has seen tremendous success in healthcare analytics, particularly for tasks in medical imaging such as diagnostics [1]. 
