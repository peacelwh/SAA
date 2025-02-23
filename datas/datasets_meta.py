import os
import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
import torch
import re

import torchvision
import numpy as np
from datas.dataloader import MultiTrans


def build_transform(image_size):
    transform = create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    return transform


class Datasets(Dataset):
    def __init__(self, root_path, split='train', **args):
        dataset_name=root_path
        
        # Set the path #
        DATASET_DIR = '/home/lvg/fsl/datasets/'+root_path
            
        if split == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'base')
        elif split == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'novel')
        elif split == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
        else:
            raise ValueError('Unkown setname.')
        
        # normalization #
        if root_path in ['miniImageNet', 'tieredImageNet','CD-CUB', 'CUB','CUB-raw','FG-Dogs','FG-Cars']:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            norm_params = {"mean": mean, 
                        "std": std}
        elif root_path=='CIFAR-FS' or root_path=='FC100':
            mean = (0.5071, 0.4866, 0.4409)
            std = (0.2009, 0.1984, 0.2023)
            norm_params = {"mean": mean, 
                        "std": std}

        else:#'CD-CUB', 'CUB','FG-CUB','CUB-raw','CD-Cars','FG-Cars
            mean = ([x / 255.0 for x in [125.3, 123.0, 113.9]])
            std = ([x / 255.0 for x in [63.0, 62.1, 66.7]])
            norm_params = {"mean": mean, 
                        "std": std}
        norm = transforms.Normalize(**norm_params)
        
        # Transform #
        image_size = args['image_size']

        if split=='train':
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                norm,
            ])
            # self.transform=transforms.Compose([
            #         transforms.Resize(image_size),
            #         transforms.CenterCrop(image_size),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         norm,
            #     ])    
            
            # self.transform = transforms.Compose([
            #     transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            #     transforms.RandomRotation(15),
            #     transforms.ToTensor(),
            #     norm,
            # ])
        else:
            self.transform = transforms.Compose([
                        transforms.Resize(int(image_size * 1.1)),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        norm,
                    ])


            if args['aug_support'] >1:
                aug = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm,
                ])
                self.transform = MultiTrans([self.transform]*3 + [aug]*(args['aug_support']-3))

        self.dataset = torchvision.datasets.ImageFolder(THE_PATH, self.transform)

        # visualization of raw images #
        def convert_raw(x):
                mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
                std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
                return x * std + mean
        self.convert_raw = convert_raw

        # class-label text #
        self.idx2text = {}
        if dataset_name == 'miniImageNet' or dataset_name == 'tieredImageNet':
            with open('datas/ImageNet_idx2text.txt', 'r') as f:
                for line in f.readlines():
                    idx, _, text = line.strip().split()
                    text = text.replace('_', ' ')
                    self.idx2text[idx] = text
        # elif dataset_name == 'FC100':
        #     with open('datas/FC100_idx2text.txt', 'r') as f:
        #         for line in f.readlines():
        #             idx, text = line.strip().split()
        #             idx = idx.strip(':')
        #             text = text.replace('_', ' ')
        #             self.idx2text[idx] = text
        elif dataset_name in ['CIFAR-FS', 'Places', 'FC100']: 
            for idx in self.dataset.classes:
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text
        elif dataset_name in ['CD-CUB', 'CUB','FG-CUB','CUB-raw']:
            for idx in self.dataset.classes:
                i, text = idx.split(".")
                text = text.replace('_', ' ')
                self.idx2text[idx] = text
        elif dataset_name in ['FG-Cars', 'CD-Cars','Places','Plantae']:
            for idx in self.dataset.classes:
                # text = idx
                text = idx.replace('_', ' ')
                self.idx2text[idx] = text
        elif dataset_name =='FG-Dogs':
            for idx in self.dataset.classes:
                # text = idx
                text = idx.split('-')[1].replace('_', ' ')
                self.idx2text[idx] = text



    def __getitem__(self, i):
        image, label = self.dataset[i]
        text = self.dataset.classes[label]
        text = self.idx2text[text]

        # text prompt: A photo of a {label}
        text = 'A photo of a ' + text
        return image, label, text


    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dataset= Datasets('CIFAR-FS', split='train')
    #print(dataset)
    print(len(dataset.dataset.classes))
