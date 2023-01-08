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


class DataReader:
    def __init__(self) -> None:
        # directory in which the videos are stored
        self.directory_images = "C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole_resized/"
        # video list in which first the path to frame is stored. Then the frame number and then the position of the bag
        self.images_list = []
        # call funktion to store each frame
        self.read_images()
        x_train, x_test = train_test_split(self.images_list, test_size=0.2)
        print(len(self.images_list), len(x_train), len(x_test))
        pickle.dump(x_train, open("C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole/frames_train.p", "wb"))
        pickle.dump(x_test, open("C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole/frames_test.p", "wb"))

    def read_images(self):
        # take directory of videos
        directory = os.fsencode(self.directory_images)
        i = 0
        number_images = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".JPG"):
                number_images += 1
                print(number_images)
                i += 1
                # read images
                img = cv2.imread(self.directory_images + filename)
                # read label
                traco_path = "C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole/annotat.json"
                with open(traco_path) as f:
                    ann = json.loads(f.read())
                    # perform file operations

                i = 0
                pos = []
                for key in ann:

                    if (key["img"][-12:] == filename):
                        print(key["img"][-12:], filename)
                        for j in key["kp-1"]:
                            pos.append((j["x"], j["y"]))
                        ann.pop(i)
                i += 1
                # create array for each id and position
                # create new path to frame
                path_to_frame = "C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole_resized_pickled/" + filename[:-4]
                # store frame
                pickle.dump(img, open(path_to_frame + ".p", "wb"))
                # store path to frame in video list
                self.images_list.append((path_to_frame + ".p", pos))
                #os.remove("C:/Users/Florian/Desktop/Job/cc_re/" + filename)
                print(filename, pos)

                # for curr_frame in range(frameCount):
                #    self.video_list.append((self.directory_images+filename[:-3]+"p",curr_frame,whatever_dict[curr_frame]))
                continue
            else:
                continue
# To create a dataset class, I need a path-file that contains all of the train paths and another one for the test-paths
# This is necessary to allow multiple dataset instances to work in parallel
if __name__ == "__main__":
    newReader = DataReader()