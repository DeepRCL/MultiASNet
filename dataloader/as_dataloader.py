#import os
from os.path import join
from random import randint
from typing import List, Dict, Union#, Optional, Callable, Iterable
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
#from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
import warnings
from random import lognormvariate
from random import seed
import torch.nn as nn
import random
from dataloader.utils import load_as_data, preprocess_as_data, fix_leakage

seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

img_path_dataset = '/workspace/data/as_tom/annotations-all.csv'
tab_path_dataset = '/workspace/data/finetune/finetuned_df.csv'
dataset_root = r"/workspace/data/as_tom"
cine_loader = 'mat_loader'

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc
# returns a function
def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader}
    return loader_lookup_table[loader_name]

def mat_loader(path):
    mat = loadmat(path)
    if 'cine' in mat.keys():    
        return loadmat(path)['cine']
    if 'cropped' in mat.keys():    
        return loadmat(path)['cropped']


label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'normal': 0.0, 'mild': 1.0, 'moderate': 1.0, 'severe': 1.0},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1},
    'tufts': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 2}
}
class_labels: Dict[str, List[str]] = {
    'binary': ['Normal', 'AS'],
    'all': ['Normal', 'Mild', 'Moderate', 'Severe'],
    'not_severe': ['Not Severe', 'Severe'],
    'as_only': ['mild', 'moderate', 'severe'],
    'mild_moderate': ['mild', 'moderate'],
    'moderate_severe': ['moderate', 'severe'],
    'tufts': ['Normal', 'Early', 'Significant']
}

    
def get_as_dataloader(config, split, mode):
    '''
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : Configuration dictionary
        follows the format of get_config.py
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    '''
    
    if mode=='train':
        flip=config['flip_rate']
        tra = True
        bsize = config['batch_size']
        show_info = False
    if mode=='ssl':
        flip=config['flip_rate']
        tra = True
        bsize = config['batch_size']
        show_info = False
    elif mode=='val':
        flip = 0.0
        tra = False
        bsize = config['batch_size']
        show_info = False
    elif mode=='test':
        flip = 0.0
        tra = False
        bsize = 1
        show_info = True
        
    if show_info:
        assert bsize==1, "To show per-data info batch size must be 1"
    if config['model'] == 'slowfast':
        fr = 32
    else:
        fr = 16
    
    # read in the data directory CSV as a pandas dataframe
    raw_dataset = pd.read_csv(img_path_dataset)
    dataset = pd.read_csv(img_path_dataset)
        
    # append dataset root to each path in the dataframe
    dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))
    view = config['view']
        
    if view in ('plax', 'psax'):
        dataset = dataset[dataset['view'] == view]
    elif view != 'all':
        raise ValueError(f'View should be plax, psax or all, got {view}')
       
    # remove unnecessary columns in 'as_label' based on label scheme
    label_scheme_name = config['label_scheme_name']
    scheme = label_schemes[label_scheme_name]
    dataset = dataset[dataset['as_label'].isin( scheme.keys() )]

    #load tabular dataset
    tab_train, tab_val, tab_test = load_as_data(csv_path = tab_path_dataset,
                                                drop_cols = config['drop_cols'],
                                                num_ex = config['num_ex'],
                                                scale_feats = config['scale_feats'])
                                                

    #perform imputation 
    train_set, val_set, test_set, all_cols = preprocess_as_data(tab_train, tab_val, tab_test, config['categorical_cols'])
    
    # Take train/test/val
    if split in ('train', 'val', 'test', 'ulb'):
        dataset = dataset[dataset['split'] == split]
        if split=='train':
            tab_dataset = train_set
        elif split=='val':
            tab_dataset = val_set
        elif split=='test':
            tab_dataset = test_set
    elif split == 'train_all':
        dataset = dataset[dataset['split'].isin(['train','ulb'])]
        tab_dataset = train_set
    elif split != 'all':
        raise ValueError(f'View should be train/val/test/all, got {split}')
    
    #Fix data leakage 
    if split == 'train':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'val':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
    elif split == 'test':
        dataset = fix_leakage(df=raw_dataset, df_subset=dataset, split=split)
        
    dset = AorticStenosisDataset(img_path_dataset=dataset, 
                                tab_dataset=tab_dataset,
                                split=split,
                                transform=tra,
                                normalize=True,
                                frames=fr,
                                return_info=show_info,
                                contrastive_method = config['cotrastive_method'],
                                flip_rate=flip,
                                label_scheme = scheme)
    
    if mode=='train':
        if config['sampler'] == 'AS':
            sampler_AS, _ = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_AS, num_workers=10)
        elif config['sampler'] == 'bicuspid':
            _ , sampler_B = dset.class_samplers()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_B, num_workers=10)
        else: # random sampling
            loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=10)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=10)
    return loader
    

class AorticStenosisDataset(Dataset):
    def __init__(self, 
                 label_scheme,
                 img_path_dataset,
                 tab_dataset,
                 split: str = 'train',
                 transform: bool = True, normalize: bool = True, 
                 frames: int = 16, resolution: int = 224,
                 cine_loader: str = 'mat_loader', return_info: bool = False, #TODO
                 contrastive_method: str = 'CE',
                 flip_rate: float = 0.3, min_crop_ratio: float = 0.8, 
                 hr_mean: float = 4.237, hr_std: float = 0.1885,
                 **kwargs):

        self.return_info = return_info
        self.hr_mean = hr_mean
        self.hr_srd = hr_std
        self.scheme = label_scheme
        self.cine_loader = get_loader(cine_loader)
        self.dataset = img_path_dataset

        #normalize age
        self.dataset['age_norm'] = (self.dataset['age'] - self.dataset['age'].mean()) / self.dataset['age'].std()

        self.tab_dataset = tab_dataset
        self.frames = frames
        self.resolution = (resolution, resolution)
        self.split = split
        self.transform = None
        self.transform_contrastive = None
        self.pack_transform = PackPathway(alpha=4)
        if transform:
            self.transform = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
            #     [RandomResizedCropVideo(size=self.resolution, scale=(min_crop_ratio, 1)),
            #      RandomHorizontalFlipVideo(p=flip_rate)]
            # )
        if contrastive_method!= 'CE':
            self.transform_contrastive = Compose(
                [RandomResizedCrop(size=self.resolution, scale=(min_crop_ratio, 1)),
                 RandomHorizontalFlip(p=flip_rate)]
            )
            #     [RandomResizedCropVideo(size=self.resolution, scale=(min_crop_ratio, 1)),
            #      RandomHorizontalFlipVideo(p=flip_rate)]
            # )
            
        self.normalize = normalize
        self.contrstive = contrastive_method

    def class_samplers(self):
        # returns WeightedRandomSamplers
        # based on the frequency of the class occurring
        
        # storing labels as a dictionary will be in a future update
        labels_B = np.array(self.dataset['Bicuspid'])*1
        # storing labels as a dictionary will be in a future update
        labels_AS = np.array(self.dataset['as_label'])  
        labels_AS = np.array([self.scheme[t] for t in labels_AS])
        class_sample_count_AS = np.array([len(np.where(labels_AS == t)[0]) 
                                          for t in np.unique(labels_AS)])
        weight_AS = 1. / class_sample_count_AS
        if len(weight_AS) != 4:
            weight_AS = np.insert(weight_AS,0,0)
        samples_weight_AS = np.array([weight_AS[t] for t in labels_AS])
        samples_weight_AS = torch.from_numpy(samples_weight_AS).double()
        #samples_weight_AS = samples_weight_AS.double()
        sampler_AS = WeightedRandomSampler(samples_weight_AS, len(samples_weight_AS))
        if labels_B[0] == labels_B[0]:
            class_sample_count_B = np.array([len(np.where(labels_B == t)[0]) 
                                             for t in np.unique(labels_B)])
            weight_B = 1. / class_sample_count_B
            samples_weight_B = np.array([weight_B[t] for t in labels_B])

            samples_weight_B = torch.from_numpy(samples_weight_B).double()
            #samples_weight_B = samples_weight_B.double()
            sampler_B = WeightedRandomSampler(samples_weight_B, len(samples_weight_B))
        else:
            sampler_B = 0
        return sampler_AS, sampler_B
        

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def get_random_interval(vid, length):
        length = int(length)
        start = randint(0, max(0, len(vid) - length))
        return vid[start:start + length]
    
    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)
    
    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor-m)/std

    def __getitem__(self, item):
        data_info = self.dataset.iloc[item]

        #get associated tabular data based on echo ID
        study_num = data_info['Echo ID#']
        tab_info = self.tab_dataset.loc[int(study_num)]
        tab_info = torch.tensor(tab_info.values, dtype=torch.float32)

        cine_original = self.cine_loader(data_info['path'])

        #if raw_dataset comes from all_cines in nas drive...
        folder = 'round2'
        if folder == 'all_cines':
            cine_original = cine_original.transpose((2,0,1))
        elif folder == 'round2':
            pass
            
        window_length = 60000 / (lognormvariate(self.hr_mean, self.hr_srd) * data_info['frame_time'])
        cine = self.get_random_interval(cine_original, window_length)
        #print(cine.shape)
        cine = resize(cine, (32, *self.resolution))
        cine = torch.tensor(cine).unsqueeze(0)
        
        # storing labels as a dictionary will be in a future update
        if folder == 'round2':
            labels_B = torch.tensor(int(data_info['Bicuspid']))
        if folder == 'all_cines':
            labels_B = torch.tensor(int(2))
        labels_AS = torch.tensor(self.scheme[data_info['as_label']])
        labels_age = torch.tensor(int(data_info['age_norm']))

        if self.transform:
            if self.contrstive == 'CE' or self.contrstive == 'Linear':
                cine = self.transform(cine)
            else:
                cine_org = self.transform(cine)
                cine_aug = self.transform_contrastive(cine)
                cine = cine_org
                if random.random() < 0.4:
                    upsample = nn.Upsample(size=(16,224, 224), mode='nearest')
                    cine_aug = cine_aug[:, :,  0:180, 40:180].unsqueeze(1)
                    cine_aug = upsample(cine_aug).squeeze(1)   
                
        if self.normalize:
            cine = self.bin_to_norm(cine)
            if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='all'):
                cine_aug = self.bin_to_norm(cine_aug)  

        cine = self.gray_to_gray3(cine)
        cine = cine.float()
        
        if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='train_all'):
            cine_aug = self.gray_to_gray3(cine_aug)
            cine_aug = cine_aug.float()
            
        
        # slowFast input transformation
        #cine = self.pack_transform(cine)
        if (self.contrstive == 'SupCon' or self.contrstive =='SimCLR') and (self.split == 'train' or self.split =='train_all'):
            ret = ([cine,cine_aug], tab_info, labels_AS, labels_B, labels_age)
       
        else:
            ret = (cine, tab_info, labels_AS, labels_B, labels_age)
        if self.return_info:
            di = data_info.to_dict()
            di['window_length'] = window_length
            di['original_length'] = cine_original.shape[1]
            ret = (cine, tab_info, labels_AS, labels_B, labels_age, di, cine_original)

        return ret

# extra transformation for slowfast architecture    
class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha=8):
        self.alpha = alpha
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list