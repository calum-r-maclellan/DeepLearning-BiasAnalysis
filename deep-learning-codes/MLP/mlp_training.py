
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignores annoying tf warning raised when loading embeddings: means nothing so get it off
import time 
import copy
import numpy as np
import pandas as pd
from skimage.io import imread
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from utils import compute_perform_stats, compute_geometry
from sklearn.metrics import roc_curve, confusion_matrix

def get_dataloaders(cfgs, split_labels_exist=False):
    

    if cfgs.age_bin is not None:
        df_metadata = pd.read_csv(os.path.join(cfgs.root_dir, cfgs.metadata_dir, 
                                               'age_domains', 'Age'+cfgs.age_bin + '_subgroup' + '.csv'))
        embed_path = os.path.join(cfgs.root_dir, cfgs.embedding_dir)
    
    elif cfgs.race_bin is not None:
        df_metadata = pd.read_csv(os.path.join(cfgs.root_dir, cfgs.metadata_dir, 
                                               'race_domains', cfgs.race_bin + '_subgroup' + '.csv'))
        embed_path = os.path.join(cfgs.root_dir, cfgs.embedding_dir)
    
    else:
        df_metadata = pd.read_csv(os.path.join(cfgs.root_dir, cfgs.metadata_dir, 'metadata_foundation_embeddings_numpy_raceLabels.csv')).drop(columns=['Unnamed: 0'])
        embed_path = os.path.join(cfgs.root_dir, cfgs.embedding_dir)


    if split_labels_exist:
        # training set
        df_train = df_metadata.loc[df_metadata['split']=='train']
        train_dataset = MimicCXRDatasetEmbeddings(df_train, embed_path)    
        train_loader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        print('Size of training set: {}'.format(len(df_train)))

        # validation set
        df_val = df_metadata.loc[df_metadata['split']=='validate']
        val_dataset = MimicCXRDatasetEmbeddings(df_val, embed_path)    
        val_loader = DataLoader(val_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        print('Size of validation set: {}'.format(len(df_val)))

        # testing set
        df_test = df_metadata.loc[df_metadata['split']=='test']
        test_dataset = MimicCXRDatasetEmbeddings(df_test, embed_path)    
        test_loader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        print('Size of testing set: {}'.format(len(df_test)))

        return {'train':train_loader, 'val':val_loader, 'test':test_loader}

    else:

        ''' Method 1: traditional train/val/test splits based on percentages 
                --> issue: have  around 175k training but only 20k val: takes too long to optimise over and risk of poor representation in validation.
        '''
        # # create dataset and assign indices for splitting into train/val/test
        # dataset = MimicCXRDatasetEmbeddings(df_metadata, embed_path)    

        # total_for_training = int(np.floor(cfgs.train_split * len(dataset))) # number for optimisation (includes both train/val)
        # num_val = int(np.floor(0.1 * total_for_training)) # reserve 10% of total training set for validation
        # num_train = total_for_training - num_val 
        # num_test = int(np.floor(len(dataset) - total_for_training))
        # # print(num_train, num_val, num_test)
        # indices = list(range(len(dataset)))

        # train_loader = DataLoader(
        #     dataset, batch_size=cfgs.batch_size, 
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:num_train]),
        #     pin_memory=True, num_workers=0
        # )

        # val_loader = DataLoader(
        #     dataset, batch_size=cfgs.batch_size, 
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[num_train:num_val]),
        #     pin_memory=True, num_workers=0
        # )


        ''' Method 2: select manually sized subsets from dataset and assign to train/val/test
        '''
        # training set)
        num_train = cfgs.num_train
        num_val = cfgs.num_val
        num_test = int(len(df_metadata) - (num_train + num_val))

        # simply selecting the first N without caring about class imbalacing (can be fixed with posWeight in BCE, but might change structure of loss function: BAD?) 
        # df_train = df_metadata.sample(num_train, random_state=cfgs.seed)
        # print(df_train.head(10))
        # df_train = df_metadata.iloc[:num_train, :] 
        # dataset should be balancing the classes 50/50: do this below
        df_train_pos = df_metadata.loc[df_metadata[cfgs.pathology_class]==1.0].sample(int(0.5*num_train), random_state=cfgs.seed)  #.sample(int(0.5*num_val))#.iloc[:int(0.5*num_train), :] # first N/2 of positive cases (0.5 because other half for negative)
        df_train_neg = df_metadata.loc[df_metadata[cfgs.pathology_class]==0.0].sample(int(0.5*num_train), random_state=cfgs.seed)#.iloc[:int(0.5*num_train), :] # first N/2 of negative cases (0.5 because other half for positive)
        
        # concatenate pos/neg samples and shuffle
        df_train = pd.concat([df_train_pos, df_train_neg], ignore_index=True)
        df_train = df_train.sample(frac = 1)

        if cfgs.age_bin is not None:
            df_train.to_csv('./{}_{}_df_train.csv'.format(cfgs.pathology_class, cfgs.age_bin))
        elif cfgs.race_bin is not None:
            df_train.to_csv('./{}_df_train.csv'.format(cfgs.race_bin))

        train_dataset = MimicCXRDatasetEmbeddings(df_train, embed_path, cfgs.pathology_class, cfgs.num_classes)    
        train_loader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        print('Size of training set: {}'.format(len(df_train)))

        # validation set
        ''' bits below for multilabel classification
        '''
        # df_val = df_metadata.loc[~df_metadata['embedding_array'].isin(df_train['embedding_array'])]
        # df_val = df_val.sample(num_val, random_state=cfgs.seed)

        # simply selecting the first N without caring about class imbalacing (can be fixed with posWeight in BCE, but might change structure of loss function: BAD?)         
        # df_val = df_metadata.iloc[num_train:(num_train+num_val), :]
        # dataset should be balancing the classes 50/50: do this below
        df_val_pos = df_metadata.loc[(df_metadata[cfgs.pathology_class]==1.0) & 
                                     (~df_metadata['embedding_array'].isin(df_train['embedding_array']))
                                    ].sample(int(0.5*num_val), random_state=cfgs.seed)#.iloc[:int(0.5*num_val), :] # return pos samples NOT in training set
    
        df_val_neg = df_metadata.loc[(df_metadata[cfgs.pathology_class]==0.0) & 
                                     (~df_metadata['embedding_array'].isin(df_train['embedding_array']))
                                    ].sample(int(0.5*num_val), random_state=cfgs.seed)#.iloc[:int(0.5*num_val), :] # return pos samples NOT in training set

        # concatenate pos/neg samples and shuffle
        df_val = pd.concat([df_val_pos, df_val_neg], ignore_index=True)
        df_val = df_val.sample(frac = 1)
        
        if cfgs.age_bin is not None:
            df_val.to_csv('./{}_{}_df_val.csv'.format(cfgs.pathology_class, cfgs.age_bin))
        elif cfgs.race_bin is not None:
            df_val.to_csv('./{}_df_val.csv'.format(cfgs.race_bin))
        
        val_dataset = MimicCXRDatasetEmbeddings(df_val, embed_path, cfgs.pathology_class, cfgs.num_classes)    
        val_loader = DataLoader(val_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        print('Size of validation set: {}'.format(len(df_val)))

        # check its TRUE that no training/validation data overlap
        assert any(~df_train['embedding_array'].isin(df_val['embedding_array'])), 'Training/validation overlap.'

        # testing set
        # df_test = df_metadata.iloc[(num_train+num_val):, :]
        # return embeddings NOT in either training or validation sets
        df_trainval_embeds = pd.concat([df_train, df_val], ignore_index=True)     
        df_test = df_metadata.loc[~df_metadata['embedding_array'].isin(df_trainval_embeds['embedding_array'])]
        
        if cfgs.age_bin is not None:
            df_test.to_csv('./{}_{}_df_test.csv'.format(cfgs.pathology_class, cfgs.age_bin))
        elif cfgs.race_bin is not None:
            df_test.to_csv('./{}_{}_df_test.csv'.format(cfgs.pathology_class, cfgs.race_bin))
        
        test_dataset = MimicCXRDatasetEmbeddings(df_test, embed_path, cfgs.pathology_class, cfgs.num_classes)    
        test_loader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
        
        # double check the merging/assignment done correctly based on expected test set size
        assert (num_test == len(df_test)), 'Test set wrong size.'
        print('Size of testing set: {}'.format(len(df_test)))

        return {'train':train_loader, 'val':val_loader, 'test':test_loader}


class MimicCXRDatasetEmbeddings(Dataset):
    def __init__(self, mimic_cxr_metadata, embed_path, pathology_class, num_classes=1):

        self.pathology = pathology_class
        self.num_classes = num_classes
        self.metadata = mimic_cxr_metadata
        self.n_positive, self.n_negative = sum(mimic_cxr_metadata[self.pathology]==1.), sum(mimic_cxr_metadata[self.pathology]==0.)
        self.pos_weight = 1 / (self.n_positive / self.n_negative)
        
        if num_classes == 1:
            self.labels = [pathology_class]
        else:    
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
            # 'Atelectasis',
            # 'Cardiomegaly',
            # 'Consolidation',
            # 'Edema',
            # 'Enlarged Cardiomediastinum',
            # 'Fracture',
            # 'Lung Lesion',
            # 'Lung Opacity',
            # 'No Finding',
            # 'Pleural Effusion',
            # 'Pleural Other',
            # 'Pneumonia',
            # 'Pneumothorax',
            # 'Support Devices']

        self.samples = []
        if self.num_classes == 1:
            # binary classification:
            # --> training a single binary classifier to predict a SINGLE pathology (eg Cardiomegaly)
            for idx, _ in self.metadata.iterrows():
                filename = self.metadata.loc[idx, 'embedding_array']
                img_label = np.array(self.metadata.loc[idx, self.pathology], dtype='float32')
                sample = {'input_path': embed_path + filename, 'pathology_label': img_label}
                self.samples.append(sample)

        else:
            # multilabel binary classification:
            # --> all pathologies present in the image, meaning separate binary classifiers are trained for each pathology
            for idx, _ in self.metadata.iterrows():
                filename = self.metadata.loc[idx, 'embedding_array']
                img_label = np.zeros(len(self.labels), dtype='float32')
                for i in range(0, len(self.labels)):
                    img_label[i] = np.array(self.metadata.loc[idx, self.labels[i].strip()] == 1, dtype='float32')
                sample = {'input_path': embed_path + filename, 'pathology_label': img_label}
                self.samples.append(sample)
          

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        sample = self.samples[index] 
        inputs = torch.from_numpy(np.load(sample['input_path']))
        lbls = torch.from_numpy(sample['pathology_label'])
        return {'input': inputs, 'label': lbls}
   

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
    

def train(model, data_loader, optimiser, criterion, device):
    model.train()
    loss_list = []
    auc_list, f1_list, prec_list, recall_list = [], [], [], []
    for i, batch in enumerate(data_loader):
        inputs, labels = batch['input'].to(device), batch['label'].to(device)
        optimiser.zero_grad()
        preds = model(inputs).squeeze()
        loss = criterion(preds, labels)
        # probs = torch.sigmoid(preds)
        # loss = F.binary_cross_entropy(probs, labels)
        loss.backward()
        optimiser.step()
        loss_list.append(loss.item())
        
        # classification metrics
        probs = torch.sigmoid(preds)

        # uncomment for binary or multi-class
        # y_preds = np.round(probs.detach().cpu().numpy()) 
        # uncomment for multi-label (eg binary label for all classes eg chexpert)
        y_preds = probs.view(-1).detach().cpu().numpy()
        y_trues = labels.view(-1).detach().cpu().numpy()

        perform_stats = compute_perform_stats(y_preds, y_trues, num_classes=model.num_classes) # uncomment for multi-class
        auc_list.append(perform_stats['auc']) 
        f1_list.append(perform_stats['f1'])
        prec_list.append(perform_stats['precision']) 
        recall_list.append(perform_stats['recall'])
    
    # summarise over epoch
    return {'loss':np.mean(loss_list), 'auc': np.mean(auc_list), 'f1':np.mean(f1_list), 'precision':np.mean(prec_list),'recall':np.mean(recall_list)}
       

def validation(model, data_loader, criterion, device, test=False):
    model.eval()
    loss_list = []
    auc_list, f1_list, prec_list, recall_list = [], [], [], []
    logits, predictions, targets = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            preds = model(inputs).squeeze()
            loss = criterion(preds, labels)
            # probs = torch.sigmoid(preds)
            # loss = F.binary_cross_entropy(probs, labels)
            loss_list.append(loss.item())

            # classification metrics
            probs = torch.sigmoid(preds)    
            # uncomment for binary or multi-class
            # y_preds = np.round(probs.detach().cpu().numpy()) 
            # uncomment for multi-label (eg binary label for all classes eg chexpert)
            y_preds = probs.view(-1).detach().cpu().numpy()
            y_trues = labels.view(-1).detach().cpu().numpy()

            perform_stats = compute_perform_stats(y_preds, y_trues, num_classes=model.num_classes) # uncomment for multi-class
            auc_list.append(perform_stats['auc']) 
            f1_list.append(perform_stats['f1'])
            prec_list.append(perform_stats['precision']) 
            recall_list.append(perform_stats['recall'])
    
            predictions.append(probs)
            targets.append(labels)
        
        # logits = torch.cat(logits, dim=0)
        predictions = torch.cat(predictions, dim=0).detach().cpu().numpy()
        targets = torch.cat(targets, dim=0).detach().cpu().numpy()
            
        # if test:     
        #     # get detection stats
        #     confusion = confusion_matrix(y_true=targets, y_pred=predictions)
        #     print(confusion)
            
    # summarise over epoch
    return {'loss':np.mean(loss_list), 'auc': np.mean(auc_list), 
            'f1':np.mean(f1_list), 'precision':np.mean(prec_list),
            'recall':np.mean(recall_list), 'predictions':predictions, 'targets':targets}                   
    
def main(cfgs):

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)
    torch.mps.manual_seed(cfgs.seed)
    
    '''
        Dataset and loaders
    '''
    # dataloaders = get_dataloaders(cfgs)

    embed_path = os.path.join(cfgs.root_dir, cfgs.embedding_dir)

    # df_train = pd.read_csv(cfgs.root_dir + cfgs.metadata_dir + 'MIMIC.train.subsample=10000.nomissingJPG.csv') 
    df_train = pd.read_csv(cfgs.root_dir + cfgs.metadata_dir + 'MIMIC.sample.train.dicoms_nomissingJPG.csv') 
    train_dataset = MimicCXRDatasetEmbeddings(df_train, embed_path, cfgs.pathology_class, cfgs.num_classes)    
    train_loader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    print('Size of training set: {}'.format(len(df_train)))
    
    # df_val = pd.read_csv(cfgs.root_dir + cfgs.metadata_dir + 'MIMIC.val.subsample=10000.nomissingJPG.csv') 
    df_val = pd.read_csv(cfgs.root_dir + cfgs.metadata_dir + 'MIMIC.sample.val.dicoms_nomissingJPG.csv') 
    val_dataset = MimicCXRDatasetEmbeddings(df_val, embed_path, cfgs.pathology_class, cfgs.num_classes)    
    val_loader = DataLoader(val_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    print('Size of validation set: {}'.format(len(df_val)))

    '''
        Setup the model
    '''
    # model = MLP(num_layers=cfgs.num_layers, num_ftrs=cfgs.num_ftrs, num_classes=cfgs.num_classes)
    # weight_dir = './model_init.pth'
    # if not os.path.exists(weight_dir): torch.save(model.state_dict(), weight_dir)

    model = MLP(num_layers=cfgs.num_layers, num_ftrs=cfgs.num_ftrs, num_classes=cfgs.num_classes)
    model = model.to(device)
    # model.load_state_dict(torch.load(weight_dir))
    # model_init = copy.deepcopy(model)

    '''
        Optimisation 
    '''
    if cfgs.optim == 'sgd':
        print('Using SGD optimiser')
        optimiser = torch.optim.SGD(model.parameters(), lr=cfgs.lr, momentum=cfgs.momentum)
    elif cfgs.optim == 'adam':
        print('Using ADAM optimiser')
        optimiser = torch.optim.Adam(model.parameters(), lr=cfgs.lr, betas=(cfgs.beta1, cfgs.beta2))

    if not cfgs.lr_scheduler:
        lr_scheduler = None
    else:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=cfgs.epochs, eta_min=0.1*cfgs.lr)

    criterion = nn.BCEWithLogitsLoss().to(device)

    '''
        Training loop: run train, validation, and testing steps
    '''
    t_start = time.time()
    # df_experiments = pd.DataFrame()
    best_auc = 0.0
    for epoch in range(cfgs.epochs):
        
        print('-' * 20)
        print('Epoch {}\n'.format(epoch+1))
        # print('lr: {:.4f}'.format(optimiser.param_groups[0]['lr']))

        # training epoch
        train_stats = train(model, train_loader, optimiser, criterion, device)
        print('Training stats:')
        print('\tLoss: {:.4f}, AUC: {:.4f}, Recall: {:.4f}'.format(train_stats['loss'], train_stats['auc'], train_stats['recall']))
        
        # compute geometry
        # eigenvalues, _, distance = compute_geometry(model, model_init, dataloaders['train'], device, n_eigen=cfgs.n_eigen)

        # validation epoch
        val_stats = validation(model, val_loader, criterion, device, test=False)
        print('Validation stats:')
        print('\tLoss: {:.4f}, AUC: {:.4f}, Recall: {:.4f}'.format(val_stats['loss'], val_stats['auc'], val_stats['recall']))
        print('time to complete: {:.2f}'.format(time.time()-t_start))
        print()

        torch.save(model.state_dict(), 'MLP-weights/mimic_MLP__valLoss={:.4f}_valAuc={:.4f}_epoch={}.pth'.format(val_stats['loss'], val_stats['auc'], epoch+1))
            
        if val_stats['auc'] > best_auc:
            best_auc = val_stats['auc']
            print('best AUC so far: {}'.format(best_auc))
            print('epoch: {}'.format(epoch+1))

        # stats_dict = ({ 'lr':[cfgs.lr], 'm':[cfgs.momentum], 'val_loss':[val_stats['loss']], 
        #                 'val_acc':[val_stats['acc']], 'val_f1':[val_stats['f1']],
        #                 'top1': [eigenvalues[0]], 'top2': [eigenvalues[1]], 'top3': [eigenvalues[2]],
        #                 'distance':[distance],
        #     })
        # results_dict = pd.DataFrame

        # df_exp = pd.DataFrame(stats_dict)
        # print(df_exp)
        # df_experiments = pd.concat([df_experiments, df_exp], ignore_index=True)  

        # if lr_scheduler is not None: lr_scheduler.step()



    # df_experiments.to_csv(os.path.join(cfgs.exp_dir, 'results_nEigen={}.csv'.format(cfgs.n_eigen)))





if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-root_dir', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/')
    parser.add_argument('-exp_dir', type=str, default='/Users/calummaclellan/Documents/PhD/experiments/mimic-cxr/')
    parser.add_argument('-embedding_dir', type=str, default='foundation-model-embeddings/files/')
    parser.add_argument('-metadata_dir', type=str, default='metadata/for-comparing-FM-DenseNet/')
    parser.add_argument('-pathology_class', type=str, default='Cardiomegaly')
    parser.add_argument('-batch_size', type=int, default=64, help='')
    parser.add_argument('-lr_scheduler', type=bool, default=False, help='')
    parser.add_argument('-lr', type=float, default=0.001, help='')
    parser.add_argument('-momentum', type=float, default=0.9, help='')
    parser.add_argument('-beta1', type=float, default=0.9, help='')
    parser.add_argument('-beta2', type=float, default=0.999, help='')
    parser.add_argument('-optim', choices=['sgd', 'adam'], default='adam', help='')
    parser.add_argument('-n_eigen', type=int, default=1, help='')
    parser.add_argument('-num_layers', type=int, default=3, help='')
    parser.add_argument('-num_ftrs', type=int, default=1376, help='number of features in the foundation classifier (Efficient-net has 1376)')
    parser.add_argument('-num_classes', type=int, default=14, help='number of pathologies to classify (14 for chexpert/mimic-cxr)')
    parser.add_argument('-seed', type=int, default=42, help='seed for rngs')
    parser.add_argument('-epochs', type=int, default=20, help='number of epochs to train the model for')
    parser.add_argument('-train_split', type=int, default=0.8, help='percentage of dataset used for training/validation')
    parser.add_argument('-num_train', type=int, default=500, help='manually-selected amount of data for training')
    parser.add_argument('-num_val', type=int, default=500, help='manually-selected amount of data for validation')

    cfgs = parser.parse_args()
    main(cfgs)
