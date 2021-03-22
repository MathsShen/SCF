import os
import os.path as osp
import glob

import pickle
import numpy as np
import imageio
from PIL import Image

import torch
#import torch.utils.data as data
from torchvision import transforms

import pickle
import cv2

import numpy as np
import numpy.random as npr

import pdb


class MS1M(torch.utils.data.Dataset):
    def __init__(self, transform, name='', train=True):
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'

        root_folder = "/home/lishen/Dataset/MS1M-ArcFace"
        self.data_folder = osp.join(root_folder, "images")
        self.pkl_folder = osp.join(root_folder, "scripts")

        with open(osp.join(self.pkl_folder, "id2numimgs.pkl"), 'rb') as f:
            self.id2numimgs = pickle.load(f)
        self.num_ids = len(self.id2numimgs)
        
        self.ids = tuple(self.id2numimgs.keys())

        list_fpath = osp.join(root_folder, "list/ms1m_list.txt")
        with open(osp.join(list_fpath), 'r') as f:
            list_items = f.readlines()

        list_items = [list_item[:-1] for list_item in list_items]
        self.list_items = npr.permutation(list_items) ######################################
        self.list_items = list_items

        self.transform = transform

    
    def __getitem__(self, index):
        # index indicates where to get
        list_item = self.list_items[index]
        splits = list_item.split()
        img_path = splits[0]
        
        face = cv2.imread(img_path)
        face = face[:, :, ::-1]
        face = Image.fromarray(face)
        #face = Image.open(img_path)
        face = self.transform(face)
           
        label = np.int32(splits[1])
        sample = {"face": face, "label": label}
        return sample


    def __len__(self):
        return len(self.list_items)


class MS1M_Embeddings(torch.utils.data.Dataset):
    def __init__(self, name='', train=True):
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'

        root_folder = "/home/lishen/Dataset/MS1M-ArcFace"

        list_fpath = osp.join(root_folder, "list/ms1m_list_with_flipped.txt")
        with open(list_fpath, 'r') as f:
            list_items = f.readlines()

        list_items = [list_item[:-1] for list_item in list_items]
        self.list_items = npr.permutation(list_items)

    def __getitem__(self, index):
        list_item = self.list_items[index]
        splits = list_item.split()
        npy_path = splits[0]

        embedding = np.load(npy_path)
        label = np.int32(splits[1])
        sample = {"embedding": embedding, "label": label}

        return sample

    def __len__(self):
        return len(self.list_items)


class MS1M_ConvfsAndEmbeddings(torch.utils.data.Dataset):
    def __init__(self, name='', train=True):
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'

        root_folder = "/home/lishen/Dataset/MS1M-ArcFace"

        convfs_list_fpath = osp.join(root_folder, "list/ms1m_iresnet100_cv2rgb_convfs_list_with_flipped.txt")
        embeds_list_fpath = osp.join(root_folder, "list/ms1m_iresnet100_cv2rgb_embeds_list_with_flipped.txt")
        with open(convfs_list_fpath, 'r') as f:
            convfs_list_items = f.readlines()
        with open(embeds_list_fpath, 'r') as f:
            embeds_list_items = f.readlines()

        convfs_list_items = [list_item[:-1] for list_item in convfs_list_items]
        embeds_list_items = [list_item[:-1] for list_item in embeds_list_items]
        ###############################################
        #convfs_list_items = convfs_list_items[::2]
        #embeds_list_items = embeds_list_items[::2]
        ###########FOR MANIFOLD ANALYSIS ONLY##########

        assert len(convfs_list_items) == len(embeds_list_items)
        permuted_indices = npr.permutation(list(range(len(convfs_list_items))))
        self.convfs_list_items = [convfs_list_items[ind] for ind in permuted_indices]
        self.embeds_list_items = [embeds_list_items[ind] for ind in permuted_indices]
        
        self.convfs_list_items = convfs_list_items
        self.embeds_list_items = embeds_list_items


    def __getitem__(self, index):
        convfs_list_item = self.convfs_list_items[index]
        embeds_list_item = self.embeds_list_items[index]

        splits = convfs_list_item.split()
        npy_path = splits[0]
        convfs = np.load(npy_path)
        label = np.int32(splits[1])
        
        splits = embeds_list_item.split()
        npy_path = splits[0]
        embedding = np.load(npy_path)
        label_ = np.int32(splits[1])

        #assert label == label_
        
        sample = {
                  "convf": convfs,
                  "embedding": embedding,
                  "label": label
                 }
        return sample

    def __len__(self):
        return len(self.convfs_list_items)


mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3


def get_MS1M():
    image_shape = (112, 112, 3)

    transformations = [
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std),
                      ]
    transform = transforms.Compose(transformations)

    train_dataset = MS1M(transform, train=True)
    test_dataset = MS1M(transform, train=False)

    return image_shape, None, train_dataset, test_dataset
