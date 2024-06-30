
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
from utils import compute_perform_stats
import itertools # for concatenating lists



class MimicCXRDatasetEmbeddings(Dataset):
    def __init__(self, mimic_cxr_metadata, embed_path, pathology_class, num_classes=1):

        self.pathology = pathology_class
        self.num_classes = num_classes
        self.metadata = mimic_cxr_metadata
        
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
                filename = self.metadata.loc[idx, 'embedding_array']
                dicom, patient =  self.metadata.loc[idx, 'dicom_id'], self.metadata.loc[idx, 'subject_id']
                img_label = np.zeros(len(self.labels), dtype='float32')
                for i in range(0, len(self.labels)):
                    img_label[i] = np.array(self.metadata.loc[idx, self.labels[i].strip()] == 1, dtype='float32')
                
                sample = {'input_path': embed_path + filename, 
                          'pathology_label': img_label,
                          'subject_id':patient,
                          'dicom_id': dicom}
                self.samples.append(sample)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        sample = self.samples[index] 
        inputs = torch.from_numpy(np.load(sample['input_path']))
        lbls = torch.from_numpy(sample['pathology_label'])
        return {'input': inputs, 'label': lbls, 'subject':sample['subject_id'], 'dicom':sample['dicom_id']}
   

class MLP(nn.Module):
    def __init__(self, num_layers, num_ftrs=1376, num_classes=14):
        super(MLP, self).__init__()
        
        self.num_ftrs = num_ftrs
        self.num_classes = num_classes
        
        if num_layers == 5:
            # CXR-MLP-5
            self.model = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )
        elif num_layers == 3:
            # CXR-MLP-3
            self.model = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        else:
            # CXR-Linear
            self.model = nn.Sequential(
                nn.Linear(num_ftrs, num_classes)
            )

    def forward(self, x):
        return self.model(x)
    

def test(model, data_loader, criterion, device):
    model.eval()
    loss_list = []
    auc_list, f1_list = [], []
    t_start = time.time()
    count = 0
    patients, dicoms = [], []
    targets, predictions = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            patients.append(batch['subject'])    
            dicoms.append(batch['dicom'])    

            preds = model(inputs)
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
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)

        patients = torch.cat(patients, dim=0)
        dicoms = list(itertools.chain.from_iterable(dicoms))
        

    # summarise over epoch
    return {'loss':np.mean(loss_list), 'auc': np.mean(auc_list), 'f1':np.mean(f1_list),
            'targets':targets.cpu().detach().numpy(), 'predictions':predictions.cpu().detach().numpy(),
            'subject_id':patients.numpy(), 'dicom_id':dicoms}
  

def main(cfgs):
 
    ''' Initialise device to load and rngs
    '''
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)
    torch.mps.manual_seed(cfgs.seed)

    embed_path = os.path.join(cfgs.data_dir, cfgs.embedding_dir)

    ''' Dataset and loaders
    ''' 
    df_test = pd.read_csv(cfgs.data_dir + cfgs.metadata_dir + 'MIMIC.resample.test.dicoms_nomissingJPG.csv') 
    df_test = df_test.sample(frac=1)
    test_dataset = MimicCXRDatasetEmbeddings(df_test, embed_path, cfgs.pathology_class, cfgs.num_classes)    
    test_loader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    print('Size of testing set: {}'.format(len(df_test)))

    ''' Setup the model and optimisation
    '''
    model = MLP(num_layers=cfgs.num_layers, num_ftrs=cfgs.num_ftrs, num_classes=cfgs.num_classes)
    model.load_state_dict(torch.load(cfgs.root_dir + 
                                        'codes/MLP-weights/' + 
                                        'mimic_MLP__valLoss=0.2422_valAuc=0.9023_epoch=17.pth'         
    ))                                
    
    model.to(device)
    print(model)
    print('#parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    ''' Inference step 
    '''
    test_stats = test(model, test_loader, criterion, device)
    print('Test stats:\n')
    print('\tLoss: {:.4f}, AUC: {:.4f}'.format(test_stats['loss'], test_stats['auc']))
    print('-' * 20)
    print()

    ''' Save predictions
    '''
    df_patients = pd.DataFrame()
    df_patients['subject_id'] = test_stats['subject_id']
    df_patients['dicom_id'] = test_stats['dicom_id']
    
    df_targets = pd.DataFrame(data=test_stats['targets'], columns=test_dataset.labels)
    df_targets[['subject_id', 'dicom_id']] = df_patients[['subject_id', 'dicom_id']]
    df_targets.to_csv(cfgs.root_dir + 'codes/MLP-weights/' + 'targets.csv')
    print(df_targets.head(10))

    df_predictions = pd.DataFrame(data=test_stats['predictions'], columns=test_dataset.labels)
    df_predictions[['subject_id', 'dicom_id']] = df_patients[['subject_id', 'dicom_id']]
    df_predictions.to_csv(cfgs.root_dir + 'codes/MLP-weights/' + 'predictions.csv')
    print(df_predictions.head(10))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-root_dir', type=str, default='/Users/calummaclellan/Documents/PhD/codes/mimic-cxr/src_embeddings/')
    parser.add_argument('-data_dir', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/')
    parser.add_argument('-exp_dir', type=str, default='/Users/calummaclellan/Documents/PhD/experiments/mimic-cxr/')
    parser.add_argument('-embedding_dir', type=str, default='foundation-model-embeddings/files/')
    parser.add_argument('-metadata_dir', type=str, default='metadata/for-comparing-FM-DenseNet/')
    parser.add_argument('-pathology_class', type=str, default='Cardiomegaly')
    parser.add_argument('-batch_size', type=int, default=64, help='')
    parser.add_argument('-num_layers', type=int, default=3, help='')
    parser.add_argument('-num_ftrs', type=int, default=1376, help='number of features in the foundation classifier (Efficient-net has 1376)')
    parser.add_argument('-num_classes', type=int, default=14, help='number of pathologies to classify (14 for chexpert/mimic-cxr)')
    parser.add_argument('-seed', type=int, default=42, help='seed for rngs')
    
    cfgs = parser.parse_args()
    main(cfgs)

