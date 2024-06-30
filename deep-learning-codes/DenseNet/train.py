
import os
import time
import numpy as np
import pandas as pd
from skimage.io import imread
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from utils import compute_perform_stats, compute_geometry

class MimicCXRDataset(Dataset):
    def __init__(self, metadata, cxr_dir):
        
        self.metadata = metadata
        self.cxr_dir = cxr_dir
        self.img_size = 224
        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.samples = []
        for idx, _ in self.metadata.iterrows():
            img_name = self.metadata.loc[idx, 'dicom_id']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.metadata.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': self.cxr_dir + img_name + '_resized.jpg', 
                      'pathology_label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.metadata)
    
    def _processImage(self, img_path):
        img = imread(img_path).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.repeat(3, 1, 1)
        return img

    def __getitem__(self, index):
        sample = self.samples[index] 
        imgs = self._processImage(sample['image_path'])
        lbls = torch.from_numpy(sample['pathology_label'])
        return {'image': imgs, 'label': lbls}
   

class DenseNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=14):
        super(DenseNet, self).__init__()
        
        self.num_classes = num_classes
        self.model = torchvision.models.densenet121(pretrained=pretrained)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, self.num_classes)
        
    def forward(self, x):
        return self.model.forward(x)
    

def train(model, data_loader, optimiser, criterion, device):
    model.train()
    loss_list = []
    auc_list, f1_list = [], []
    t_start = time.time()
    count = 0
    for i, batch in enumerate(data_loader):
        
        inputs, labels = batch['image'].to(device), batch['label'].to(device)
        
        # optimisation steps
        optimiser.zero_grad()
        preds = model(inputs)
        # loss = F.binary_cross_entropy(probs, labels)
        loss = criterion(preds, labels)
        loss.backward()
        optimiser.step()
        loss_list.append(loss.item())
        
        # classification metrics
        probs = torch.sigmoid(preds)
        y_probs = probs.view(-1).detach().cpu().numpy()
        y_trues = labels.view(-1).detach().cpu().numpy()
        perform_stats = compute_perform_stats(y_probs, y_trues) # uncomment for multi-class
        auc_list.append(perform_stats['auc']) 
        f1_list.append(perform_stats['f1'])  

        count += inputs.size(0)
    
        print('{}/{}'.format(count, len(data_loader.dataset)))
        print('time: {:.2f} s'.format(time.time()-t_start))
    
    # summarise over epoch
    return {'loss':np.mean(loss_list), 'auc': np.mean(auc_list), 'f1':np.mean(f1_list)}

def validation(model, data_loader, criterion, device):
    model.eval()
    loss_list = []
    auc_list, f1_list = [], []
    t_start = time.time()
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            # loss = F.binary_cross_entropy(probs, labels)
            loss_list.append(loss.item())

            # classification metrics
            probs = torch.sigmoid(preds)
            y_probs = probs.view(-1).detach().cpu().numpy()
            y_trues = labels.view(-1).detach().cpu().numpy()
            perform_stats = compute_perform_stats(y_probs, y_trues) # uncomment for multi-class
            auc_list.append(perform_stats['auc']) 
            f1_list.append(perform_stats['f1'])     
            
            count += inputs.size(0)
            if (i+1) % 50 == 0:
                print('{}/{}'.format(count, len(data_loader.dataset)))
                print('time: {:.2f} s'.format(time.time()-t_start))
    
    # summarise over epoch
    return {'loss':np.mean(loss_list), 'auc': np.mean(auc_list), 'f1':np.mean(f1_list)}


def main(cfgs):

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)
    torch.mps.manual_seed(cfgs.seed)

    '''
    Dataset and loaders
    '''    
    # apparently some images are missing
    # here, we find the files in preprocess_224x224_all that arent in metadata, and label as missing.
    # only load images in metadata.
    df_train_not_missing = pd.read_csv(cfgs.root_dir + 'data/train_dicoms_that_exist.csv')
    df_val_not_missing = pd.read_csv(cfgs.root_dir + 'data/val_dicoms_that_exist.csv')
    df_test_not_missing = pd.read_csv(cfgs.root_dir + 'data/test_dicoms_that_exist.csv')

    # training 
    df_train = pd.read_csv(cfgs.root_dir + 'data/MIMIC.sample.train.csv')
    shape1 = df_train.shape
    # some imgs missing in folder. line below ensures we dont try to read them during training ('FileNotFoundError')
    df_train = df_train.loc[df_train['dicom_id'].isin(df_train_not_missing['dicom_id'])]
    print('#train lost: {}'.format(shape1[0] - df_train.shape[0]))
    df_train.to_csv(cfgs.root_dir + 'data/MIMIC.sample.train.dicoms_nomissingJPG.csv')

    # N = 50000
    # df_train = df_train.sample(n=N)
    # df_train.to_csv(cfgs.root_dir + 'data/MIMIC.train.subsample={}.nomissingJPG.csv'.format(N))
    df_train = df_train.sample(frac=1)
    train_dataset = MimicCXRDataset(df_train, cfgs.cxr_dir)    
    train_loader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    print('Size of training set: {}'.format(len(df_train)))

    # validation
    df_val = pd.read_csv(cfgs.root_dir + 'data/MIMIC.sample.val.csv')
    shape1 = df_val.shape
    df_val = df_val.loc[df_val['dicom_id'].isin(df_val_not_missing['dicom_id'])]
    print('#val lost: {}'.format(shape1[0] - df_val.shape[0]))
    df_val.to_csv(cfgs.root_dir + 'data/MIMIC.sample.val.dicoms_nomissingJPG.csv')

    # df_val = df_val.sample(n=N)
    # df_val.to_csv(cfgs.root_dir + 'data/MIMIC.val.subsample={}.nomissingJPG.csv'.format(N))
    df_val = df_val.sample(frac=1)
    val_dataset = MimicCXRDataset(df_val, cfgs.cxr_dir)    
    val_loader = DataLoader(val_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    print('Size of validation set: {}'.format(len(df_val)))

    # # test
    # df_test = pd.read_csv(cfgs.root_dir + 'data/MIMIC.resample.test.csv')
    # shape1 = df_test.shape
    # df_test = df_test.loc[df_test['dicom_id'].isin(df_test_not_missing['dicom_id'])]
    # print('#test lost: {}'.format(shape1[0] - df_test.shape[0]))
    # df_test.to_csv(cfgs.root_dir + 'data/MIMIC.resample.test.dicoms_nomissingJPG.csv')

    # df_test = df_test.sample(frac=1)
    # test_dataset = MimicCXRDataset(df_test, cfgs.cxr_dir)    
    # test_loader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    # print('Size of testing set: {}'.format(len(df_test)))

    '''
    Setup the model and optimisation
    '''
    model = DenseNet(pretrained=True, num_classes=cfgs.num_classes)
    model.to(device)
    print('#parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

    optimiser = optim.Adam(model.parameters(), lr=cfgs.lr, betas=(cfgs.beta1, cfgs.beta2))
    criterion = nn.BCEWithLogitsLoss().to(device)

    '''
    Loop over entire dataset and generate features (embeddings) 
    '''
    t_training = time.time()

    for epoch in range(cfgs.epochs): 
        
        t_epoch = time.time()
        print('-' * 20)
        print('Epoch {}'.format(epoch+1))

        # training epoch
        train_stats = train(model, train_loader, optimiser, criterion, device)
        print('Training stats:\n')
        print('\tLoss: {:.4f}, AUC: {:.4f}'.format(train_stats['loss'], train_stats['auc']))

        # validation epoch
        val_stats = validation(model, val_loader, criterion, device)
        print('Validation stats:\n')
        print('\tLoss: {:.4f}, AUC: {:.4f}'.format(val_stats['loss'], val_stats['auc']))

        torch.save(model.state_dict(), 
                # 'mimic_densenet_subsampledTrainValDataset={}_valLoss={:.4f}_valAuc={:.4f}_epoch={}.pth'.format(N, val_stats['loss'], val_stats['auc'], epoch+1))
                'mimic_densenet_epoch={}.pth'.format(epoch+1))
                # 'mimic_densenet_valLoss={:.4f}_valAuc={:.4f}_epoch={}.pth'.format(val_stats['loss'], val_stats['auc'], epoch+1))

        print('time to run epoch...{} s'.format(time.time() - t_epoch))

    print('Training done.') 
    print('Time elapsed: {} hrs'.format((time.time() - t_training)/3600))

    
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-root_dir', type=str, default='/Users/calummaclellan/Documents/PhD/codes/mimic-cxr/src_images/')
    parser.add_argument('-cxr_dir', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/preproc_224x224_all/')
    parser.add_argument('-batch_size', type=int, default=64, help='')
    parser.add_argument('-lr', type=float, default=0.001, help='')
    parser.add_argument('-momentum', type=float, default=0.9, help='')
    parser.add_argument('-beta1', type=float, default=0.9, help='')
    parser.add_argument('-beta2', type=float, default=0.999, help='')
    parser.add_argument('-optim', choices=['sgd', 'adam'], default='adam', help='')
    parser.add_argument('-num_ftrs', type=int, default=1024, help='number of features in the foundation classifier (Densenet 121)')
    parser.add_argument('-num_classes', type=int, default=14, help='number of pathologies to classify (14 for chexpert/mimic-cxr)')
    parser.add_argument('-seed', type=int, default=42, help='seed for rngs')
    parser.add_argument('-epochs', type=int, default=20, help='number of epochs to train the model for')
    
    cfgs = parser.parse_args()

    main(cfgs)
