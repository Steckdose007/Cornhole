import posixpath
import json
import cv2
import os
import pickle
import pathlib
import json
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import Dataset
import torch
from torchvision import transforms, datasets, models
from torchvision.models import resnet18, ResNet18_Weights
import math
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models




class TracoDataset(Dataset):
    # load the pickl lists
    def __init__(self, mode, transform=None):
        if mode == 'train':
            self.samples = pickle.load(open("C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole/frames_train.p", "rb"))
        elif mode == 'test':
            self.samples = pickle.load(open("C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole/frames_test.p", "rb"))
        else:
            raise ValueError(f'Selected mode {mode} is not implemented')
        pass
        self.transform = transform

    # needed for dataloader
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # take frame
        frame = pickle.load(open(self.samples[idx][0], "rb"))
        # convert to RGB
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize frame
        # im_rgb = cv2.resize(im_rgb, dsize=(256,256))

        frameWidth, frameHeight, channels = frame.shape
        # Creation of target
        pos_list = self.samples[idx][1]

        #create label
        base = np.zeros([256, 256], dtype=np.uint8)
        for elem in pos_list:
            # reduziert die pos auf 256,256 bild
            pos_x = (elem[0] / 100) * 256
            pos_y = (elem[1] / 100) * 256
            # print(pos_x,pos_y)
            RADIUS = 4
            base = cv2.circle(base, (int(pos_x), int(pos_y)), RADIUS, 100, -1)

        im_rgb = self.transform(im_rgb)
        base = self.transform(base)
        #print(type(im_rgb), type(base))
        return im_rgb, base


if __name__ == "__main__":
    from torchvision import transforms

    train_transform = transforms.Compose([transforms.ToTensor()])
    print(torch.cuda.is_available())
    some_dataset = TracoDataset(mode='test', transform=train_transform)
    img, target = some_dataset.__getitem__(173)
    fig, axs = plt.subplots(2)
    target = torch.squeeze(target)
    target = target.numpy()
    axs[0].imshow(target, interpolation='nearest')
    img = torch.squeeze(img)
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    axs[1].imshow(img, interpolation='nearest')
    plt.show()