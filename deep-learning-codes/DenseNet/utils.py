

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from PyHessian import *
import torch
import torch.nn as nn
import os 
from argparse import ArgumentParser


def save_embeddings(data, embed_dir):
    df = pd.read_csv(embed_dir + 'mimic_cxr_densenet_embeddings.csv')
    save_dir = embed_dir + 'files/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for idx, row in df.iterrows():
        emb = data[idx, :]
        dicom = row['dicom_id']
        fname = str(dicom) + '_densenet.npy'
        np.save(os.path.join(save_dir, fname), emb)
        if (idx % 5000) == 0:
            print('\t{}/{}...'.format(idx, data.shape[0]))
        

def compute_perform_stats(preds, labels, n_classes=14, thr=0.5):
    
    # accuracy = accuracy_score(labels, preds)
    precisions = precision_score(labels, preds > thr, average=None,
                                 labels=range(n_classes), zero_division=0.)
    recalls = recall_score(labels, preds > thr, average=None, labels=range(n_classes),
                           zero_division=0.)
    f1 = f1_score(labels, preds > thr, average=None, labels=range(n_classes),
                           zero_division=0.)
    auc_score = roc_auc_score(labels, preds)

    return {'auc':auc_score, 'precision': precisions, 'recall': recalls, 'f1': f1}

def check_missing_jpg(metadata, cfgs):
    '''
        Function for searching metadata for missing files in mimic-cxr (no idea why theyve gone) 
    '''
    dicoms = []
    counter = 0
    for idx, row in metadata.iterrows():

        img_name = row['dicom_id']
        img_path = cfgs.cxr_dir + img_name + '_resized.jpg'

        if not os.path.exists(img_path):
            print('File {} missing. Move to next...'.format(img_name))
            counter+=1
            continue
        else:
            dicoms.append(img_name)
    print(counter)
    print('#remaining: {}'.format(len(metadata) - counter))
    df = pd.DataFrame({'dicom_id': dicoms})
    df.to_csv(cfgs.root_dir + 'data/{}_dicoms_that_exist.csv'.format(cfgs.subset))



def compute_geometry(model, model_init, train_loader, device='cuda'):
    pdist = nn.PairwiseDistance(p=2)
    criterion = nn.MSELoss()
    model.eval()

    # distance of parameters from init
    params1 = torch.cat([p.view(-1) for p in model_init.parameters()])
    params2 = torch.cat([p.view(-1) for p in model.parameters()])
    dist = pdist(params1, params2)
    # print('\tparameter distance: {}'.format(dist))

    # hessian
    Hessian = hessian(model, criterion, dataloader=train_loader)
    eigenvalues, eigenvector = Hessian.eigenvalues(top_n=1)
    # print('\ttop eigenvalue: {}'.format(eigenvalues[0]))

    return eigenvalues[0], eigenvector[0], dist.item()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-root_dir', type=str, default='/Users/calummaclellan/Documents/PhD/codes/mimic-cxr/src_images/')
    parser.add_argument('-subset', type=str, default='train') # (train=293, val=156, test=1007)
    parser.add_argument('-cxr_dir', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/preproc_224x224_all/')
    cfgs = parser.parse_args()

    # metadata = pd.read_csv(cfgs.root_dir + 'data/MIMIC.resample.{}.csv'.format(cfgs.subset))
    # check_missing_jpg(metadata, cfgs)

    embed_dir = cfgs.root_dir + 'DenseNet-MIMIC-CXR-2ndMay24/' + 'embeddings/'
    data = np.load(embed_dir + 'embeddings_densenet.npy')
    print('embeddings loaded: shape of {}'.format(np.shape(data)))
    print('renaming embeddings according to dicom_id...')
    save_embeddings(data, embed_dir)
    print('...done!')
