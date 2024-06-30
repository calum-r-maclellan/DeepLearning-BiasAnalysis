
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
import itertools # for concatenating lists
from train import validation


def get_embeddings(model, input):
    # get features from output of penultimate conv layer 
    feats = model.model.features(input)
    # .size() = [B, 1024, 7, 7]

    # push through relu 
    feats = F.relu(feats, inplace=True)

    # avg pool over [B, 1024, 7, 7] to give [B, 1024, 1, 1]: kernel is 7x7, but use adaptive pool to figure this out automat
    feats = F.adaptive_avg_pool2d(feats, (1, 1))
     # .size() = [B, 1024, 1, 1]

    # reduce dims for matrix mult with linear weights (for classif output)
    embeds = feats.squeeze(2).squeeze(2)
    # embeds = torch.flatten(feats, 1)
    # .size() = [B, 1024]
    return embeds

def save_embeddings(data, save_dir):
    seq = 0
    subjects, embed_names = [], []
    df_embeds = pd.DataFrame()
    n = np.shape(data['embeddings'])[0]
    for i in range(n):
        seq += 1
        emb, subj, dicom = data['embeddings'][i, :], data['subject_id'][i], data['dicom_id'][i], 
        # name embedding after the original dicom_id (.jpg image) to be consistent with names.
        fname = str(dicom) + '_densenet.npy'
        np.save(os.path.join(save_dir, fname), emb)
        embed_names.append(fname)
        subjects.append(subj)

    df_embeds['subject_id'] = subjects
    df_embeds['embedding_array'] = embed_names
    df_embeds.to_csv(os.path.join(save_dir, 'mimic_cxr_densenet_embeddings.csv'))

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
            patient = self.metadata.loc[idx, 'subject_id']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.metadata.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': self.cxr_dir + img_name + '_resized.jpg', 
                      'pathology_label': img_label, 
                      'subject_id':patient,
                      'dicom_id': img_name}
            
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
        return {'image': imgs, 'label': lbls, 'subject':sample['subject_id'], 'dicom':sample['dicom_id']}
   

class DenseNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=14):
        super(DenseNet, self).__init__()
        
        self.num_classes = num_classes
        self.model = torchvision.models.densenet121(pretrained=pretrained)
        self.num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.num_ftrs, self.num_classes)
        
    def forward(self, x):
        return self.model.forward(x)


def test(model, data_loader, criterion, device, save_dir):
    model.eval()
    loss_list = []
    auc_list, f1_list = [], []
    t_start = time.time()
    count = 0
    embeddings, patients, dicoms = [], [], []
    targets, predictions = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            inputs, labels = batch['image'].to(device), batch['label'].to(device)

            # get embeddings + subject_ids
            features = get_embeddings(model, inputs)
            embeddings.append(features)
            patients.append(batch['subject'])    
            dicoms.append(batch['dicom'])    

            # continue with predictions
            preds = model.model.classifier(features)
            loss = criterion(preds, labels)
            loss_list.append(loss.item())

            # classification metrics
            probs = torch.sigmoid(preds)
            y_probs = probs.view(-1).detach().cpu().numpy()
            y_trues = labels.view(-1).detach().cpu().numpy()
            perform_stats = compute_perform_stats(y_probs, y_trues) # uncomment for multi-class
            auc_list.append(perform_stats['auc']) 
            f1_list.append(perform_stats['f1'])     
            targets.append(labels)
            predictions.append(probs)
            
            count += inputs.size(0)
            print('{}/{}'.format(count, len(data_loader.dataset)))
            print('time: {:.2f} s'.format(time.time()-t_start))

        # concatenate across all minibatches to create the entire dataset
        embeddings = torch.cat(embeddings, dim=0) # (N, 1024)
        # print(embeddings.size())
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)

        patients = torch.cat(patients, dim=0)
        dicoms = list(itertools.chain.from_iterable(dicoms))
        df = pd.DataFrame()
        df['subject_id'] = patients.numpy()
        df['dicom_id'] = dicoms
        df.to_csv(save_dir + 'mimic_cxr_densenet_embeddings.csv')
        # save embeddings to (N, 1024) array
        fname = save_dir + 'embeddings_densenet.npy'
        np.save(fname, embeddings.cpu().detach().numpy())

    # summarise over epoch
    return {'loss':np.mean(loss_list), 'auc': np.mean(auc_list), 'f1':np.mean(f1_list),
            'targets':targets.cpu().detach().numpy(), 'predictions':predictions.cpu().detach().numpy(),
            'subject_id':patients.numpy(), 'dicom_id':dicoms}
           
def find_best_model(dataloader, device):
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    best_auc = 0.0
    best_epoch = 1
    for i in range(10):
        
        k = i + 1
        print('Model {}:'.format(k))

        model = DenseNet(pretrained=True, num_classes=cfgs.num_classes)
        model.load_state_dict(torch.load(cfgs.root_dir + 
                                        'DenseNet-MIMIC-CXR/May11th/' +
                                        'weights/' + 
                                        'mimic_densenet_epoch={}.pth'.format(k)
                                        ))
        model = model.to(device)
        val_stats = validation(model, dataloader, criterion, device)
        print('\tValidation stats:\n')
        print('\tLoss: {:.4f}, AUC: {:.4f}'.format(val_stats['loss'], val_stats['auc']))
        if val_stats['auc'] > best_auc:
            best_auc = val_stats['auc']
            best_epoch = k

    print('best model is epoch {}, with AUC of {:.4f}'.format(best_epoch, best_auc))
    # return best_epoch


def main(cfgs):
 
    ''' Initialise device to load and rngs
    '''
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)
    torch.mps.manual_seed(cfgs.seed)

    ''' Find the best model
    '''
    if cfgs.find_best_model:
        df_val = pd.read_csv(cfgs.root_dir + 'data/MIMIC.sample.val.dicoms_nomissingJPG.csv')
        df_val = df_val.sample(frac=1)
        val_dataset = MimicCXRDataset(df_val, cfgs.cxr_dir)    
        val_loader = DataLoader(val_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        print('Size of validation set: {}'.format(len(df_val)))
        find_best_model(val_loader, device)

    ''' Dataset and loaders
    '''   
    if cfgs.run_test: 
        df_test = pd.read_csv(cfgs.root_dir + 'data/MIMIC.resample.test.dicoms_nomissingJPG.csv', low_memory=False)
        test_dataset = MimicCXRDataset(df_test, cfgs.cxr_dir)    
        test_loader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        print('Size of testing set: {}'.format(len(df_test)))

        ''' Setup the model and optimisation
        '''
        model = DenseNet(pretrained=True, num_classes=cfgs.num_classes)
        model.load_state_dict(torch.load(cfgs.root_dir + 
                                        'DenseNet-MIMIC-CXR/May10th/' +
                                        'weights/' + 
                                         'mimic_densenet_allTrainValData_valLoss=0.2595_valAuc=0.8844_epoch=6.pth'
                                        # 'mimic_densenet_epoch=6.pth'
                                        ))
        model.to(device)
        # print(model)
        print('#parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
        criterion = nn.BCEWithLogitsLoss().to(device)
        
        ''' Inference step + retrieving embeddings + saving 
        '''
        embed_dir = cfgs.root_dir + 'DenseNet-MIMIC-CXR/May10th/' + 'embeddings_epoch=6/'
        if not os.path.exists (embed_dir): os.makedirs(embed_dir)

        test_stats = test(model, test_loader, criterion, device, save_dir=embed_dir)
        print('Test stats:\n')
        print('\tLoss: {:.4f}, AUC: {:.4f}'.format(test_stats['loss'], test_stats['auc']))
        print('-' * 20)
        print()

        ''' TODO: Save predictions
        '''
        df_patients = pd.DataFrame()
        df_patients['subject_id'] = test_stats['subject_id']
        df_patients['dicom_id'] = test_stats['dicom_id']
        
        df_targets = pd.DataFrame(data=test_stats['targets'], columns=test_dataset.labels)
        df_targets[['subject_id', 'dicom_id']] = df_patients[['subject_id', 'dicom_id']]
        df_targets.to_csv(embed_dir + 'targets.csv')
        print(df_targets.head(10))

        df_predictions = pd.DataFrame(data=test_stats['predictions'], columns=test_dataset.labels)
        df_predictions[['subject_id', 'dicom_id']] = df_patients[['subject_id', 'dicom_id']]
        df_predictions.to_csv(embed_dir + 'predictions.csv')
        print(df_predictions.head(10))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-root_dir', type=str, default='/Users/calummaclellan/Documents/PhD/codes/mimic-cxr/src_images/')
    parser.add_argument('-cxr_dir', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/preproc_224x224_all/')
    parser.add_argument('-batch_size', type=int, default=64, help='')
    parser.add_argument('-num_ftrs', type=int, default=1024, help='number of features in the foundation classifier (Densenet 121)')
    parser.add_argument('-num_classes', type=int, default=14, help='number of pathologies to classify (14 for chexpert/mimic-cxr)')
    parser.add_argument('-seed', type=int, default=42, help='seed for rngs')
    parser.add_argument('-find_best_model', type=bool, default=False, help='choice to search for best model if not already done during training.')
    parser.add_argument('-run_test', type=bool, default=False, help='run test step and retrieve embeddings.')
    
    cfgs = parser.parse_args()

    main(cfgs)
