
'''
This file is for loading the raw images from MIMIC-CXR, resizing them using transforms, 
and saving again to a separate folder. 
Use minibatching with dataloaders for speedups.

@date: 29.03.2024
'''


import os,sys
from tqdm import tqdm
# sys.path.insert(0,"..")
import time
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from skimage.io import imread, imsave
from skimage.transform import resize
import pprint
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision, torchvision.transforms
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv
from PIL import Image

class MIMICDataset(Dataset):
    def __init__(self, cfgs, preproc_dir):

        self.metadata = pd.read_csv(cfgs.metadata_file)#.iloc[45000:, :]
        self.samples = []
        for _, row in self.metadata.iterrows():
            sample = {'img_path': row['img_filename'], 
                      'out_name':preproc_dir + row['dicom_id'] + '_resized.jpg'
                      }
            self.samples.append(sample)  
        self.img_size = (224, 224)
       
    def __len__(self):
        return len(self.metadata)
    
    # def _readSample(self, img_path, out_path):
    #     img = imread(img_path)
    #     img = resize(img, output_shape=self.img_size, preserve_range=True)
    #     imsave(out_path, img.astype(np.uint8))
    #     return img
    
    def _readSample(self, img_path, out_path):
        ''' running with Pillow is 6x faster the skimage!!
        '''
        try:
            img = Image.open(img_path)
            img = img.resize(self.img_size)
            img.save(out_path)
            return np.array(img)
        
        except OSError:
            print('OSError. Skipping...')
            return np.ones(self.img_size)

    def __getitem__(self, index):
        batch = self.samples[index] 
        # return {'image_path':batch['img_path'], 'out_path':batch['out_name']}
        return {'image':self._readSample(batch['img_path'], batch['out_name'])}

def resize_images(data_loader):
    t_start = time.time()
    for idx, _ in enumerate(data_loader):
        if (idx%2)==0:
            print('#images resized: {}'.format(idx*data_loader.batch_size))
            print('time elapsed: {} min'.format((time.time()-t_start)/60))
        

def main(cfgs):

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    np.random.seed(cfgs.seed)
    torch.manual_seed(cfgs.seed)
    torch.mps.manual_seed(cfgs.seed)

    '''
    Dataset and loader
    '''
    # start resizing
    preproc_dir = cfgs.out_dir
    if not os.path.exists(preproc_dir): os.makedirs(preproc_dir)

    dataset = MIMICDataset(cfgs, preproc_dir)
    data_loader = torch.utils.data.DataLoader(dataset, cfgs.batch_size, 
                                              shuffle=False, pin_memory=False,
                                              num_workers=cfgs.num_workers)
    
    # t_start = time.time()
    # batch = next(iter(data_loader))
    # # imgs, out_path = batch['image'], batch['out_path']
    # print(time.time()-t_start)

    # having the dataloader NOT shuffling is crucial for matching the embeds.npy with the demographics in the original metadata file.
    # otherwise, we'd have no way of knowing who the embedding belongs to!
    
    t_start = time.time()
    resize_images(data_loader)                  
    print('Done.')
    print('Total time elapsed: {} min'.format((time.time()-t_start)/60))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-data_harddrive_dir', type=str, default='/Volumes/MEDICALDATA/physionet.org/files/mimic-cxr-jpg/2.0.0/files/') 
    parser.add_argument('-metadata_file', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/metadata_mimic_cxr_AP_nomissingjpgs.csv')  
    # parser.add_argument('-metadata_file', type=str, default='for-pytorch/mimic_cxr_AP_pytorch.csv')  
    parser.add_argument('-out_dir', type=str, default='/Users/calummaclellan/Documents/PhD/data/mimic-cxr/preproc_224x224_more/')
    parser.add_argument('-batch_size', type=int, default=200, help='')
    parser.add_argument('-num_workers', type=int, default=0, help='')
    parser.add_argument('-seed', type=int, default=42, help='')
    
    cfgs = parser.parse_args()
    main(cfgs)




