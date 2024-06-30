

import os 
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from PyHessian import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from scipy import stats

def compute_r2(x, y):
        return stats.pearsonr(x, y)[0] 

def save_embeddings(embed_dir):
    data = np.load(embed_dir)
    n = np.shape(data['embeddings'])[0]
    for i in range(n):
        emb, dicom = data['embeddings'][i, :],  data['dicom_id'][i], 
        # name embedding after the original dicom_id (.jpg image) to be consistent with names.
        fname = str(dicom) + '_densenet.npy'
        np.save(os.path.join(embed_dir, fname), emb)
        # embed_names.append(fname)
        # subjects.append(subj)

    # df_embeds['subject_id'] = subjects
    # df_embeds['embedding_array'] = embed_names
    # df_embeds.to_csv(os.path.join(save_dir, 'mimic_cxr_densenet_embeddings.csv'))
 

def bin_age(age):
    if age <= 20:
        return 0
    if age <= 30:
        return 1
    if age <= 40:
        return 2
    if age <= 50:
        return 3
    if age <= 60:
        return 4
    if age <= 70:
        return 5
    if age <= 80:
        return 6
    return 7

def bin_label(age):
    if age <= 20:
        return '20'
    if age <= 30:
        return '30'
    if age <= 40:
        return '40'
    if age <= 50:
        return '50'
    if age <= 60:
        return '60'
    if age <= 70:
        return '70'
    if age <= 80:
        return '80'
    return '90'


def compute_grads_each_class(model, dataloader):
    model.eval()
    params = [p for p in model.parameters()]

    # first classifier [0]: atelectasis 
    weights0 = params[0][0, :]
    bias0 = params[1][0]

    inputs, labels = next(iter(dataloader))
    output = inputs*weights0 + bias0
    probs = torch.sigmoid(output)
    loss = F.binary_cross_entropy(probs, labels)
    # grad = ag.grad(loss, )


def compute_perform_stats(preds, labels, num_classes=14):
    
    ''' Assuming a 0.5 threshold (to separate 0 from 1 pred_targets), compute performance metrics
    '''
    thr = 0.5
    # accuracy = accuracy_score(labels, preds)
    # precisions = precision_score(labels, preds > thr, average=None, labels=range(num_classes), zero_division=0.)
    # recalls = recall_score(labels, preds > thr, average=None, labels=range(num_classes), zero_division=0.)
    # f1 = f1_score(labels, preds > thr, average=None, labels=range(num_classes), zero_division=0.)
    precisions = precision_score(labels, preds>thr, average=None, labels=range(num_classes), zero_division=0.)
    recalls = recall_score(labels, preds>thr, average=None, labels=range(num_classes), zero_division=0.)
    f1 = f1_score(labels, preds>thr, average=None, labels=range(num_classes), zero_division=0.)
    
    # Reminder of AUC: AUC provides an aggregate measure of performance across all possible classification thresholds. 
    # One way of interpreting AUC is as the probability that the model ranks a random positive example more highly 
    # than a random negative example. 

    auc_score = roc_auc_score(labels, preds)

    # perform_stats = {'acc':accuracy, 'precision': precisions, 'recall': recalls, 'f1': f1}
    perform_stats = {'auc': auc_score, 'precision': precisions, 'recall': recalls, 'f1': f1}
    
    # detection rates (true and false) for confusion matrix:
    # roc_stats = roc_curve(labels, preds)

    return perform_stats
    # return perform_stats, roc_stats

def compute_geometry(model, model_init, train_loader, device, n_eigen):
    pdist = nn.PairwiseDistance(p=2)
    criterion = nn.BCEWithLogitsLoss().to(device)
    model.eval()

    # distance of parameters from init
    params1 = torch.cat([p.view(-1) for p in model_init.parameters()])
    params2 = torch.cat([p.view(-1) for p in model.parameters()])
    dist = pdist(params1, params2)
    # print('\tparameter distance: {}'.format(dist))

    # hessian
    Hessian = hessian(model, criterion, dataloader=train_loader)
    eigenvalues, eigenvector = Hessian.eigenvalues(top_n=n_eigen)
    # print(eigenvalues)
    # print('\ttop eigenvalue: {}'.format(eigenvalues[0]))

    return eigenvalues, eigenvector[0], dist.item()

# if __name__ == '__main__':

    # embed_dir = 