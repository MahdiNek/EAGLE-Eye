import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import I3D.videotransforms as videotransforms


import numpy as np

from I3D.pytorch_i3d import InceptionI3d

from I3D.charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='rgb', root='/home/mahdiar/Projects/pytorch-i3d/AQA_Dataset/Figure_skating',
        split='/home/mahdiar/Projects/pytorch-i3d/charades/charades_Fake.json', batch_size=1, load_model=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data
            print(np.shape(inputs))

            b,c,t,h,w = inputs.shape

            if t > 8*128:
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                features_m = features.squeeze(0).permute(1,2,3,0).data.cpu().numpy()
                #features_m = F.upsample(features_m, 8*128, mode='linear')
                print(np.shape(features_m))
            else:
                # wrap them in Variable
                print("1111")
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                features_m = features.squeeze(0).permute(1,2,3,0)
                features_m = features_m.data.cpu().numpy()
                print(np.shape(features_m))


if __name__ == '__main__':
    # need to add argparse

    pretrained_charades_root = '/home/mahdiar/Projects/pytorch-i3d/models/rgb_charades.pt'
    root = '/home/mahdiar/Projects/pytorch-i3d/charades/Fake'

    run(mode='rgb', root= root, load_model=pretrained_charades_root)