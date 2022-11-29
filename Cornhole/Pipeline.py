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
from TracoDataset import *
import Model
from DataReader import *


def get_test_images():
 from torchvision import transforms
 train_transform = transforms.Compose([transforms.ToTensor()])
 some_dataset = TracoDataset(transform=train_transform, mode='test')
 samples = pickle.load(open("/content/drive/MyDrive/Cornhole/frames_train.p", "rb"))
 from torchvision import transforms
 train_transform = transforms.Compose([transforms.ToTensor()])
 prediction = []
 for i in range(1):
  i = i + 10
  img = pickle.load(open(samples[i][0], "rb"))
  pos_tupel = samples[i][1]
  # convert to RGB
  print("img:", type(img))
  im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  im_rgb = train_transform(im_rgb)
  print("im_rgb:", type(im_rgb))
  img = torch.unsqueeze(im_rgb, 0).to(device)
  print("im2:", type(img))
  some_result = model(img)
  some_result = torch.squeeze(some_result, 0)
  some_result = torch.squeeze(some_result, 0)
  f1 = plt.figure(1)
  plt.imshow(some_result.cpu().detach().numpy(), interpolation='nearest')

  f3 = plt.figure(3)
  im_rgb = torch.squeeze(im_rgb)
  im_rgb = im_rgb.numpy()
  im_rgb = np.transpose(im_rgb, (1, 2, 0))
  plt.imshow(im_rgb, interpolation='nearest')
  plt.show()

  # posliste auf 256 bild converten
  pos_list = []
  for coordinate in pos_tupel:
   x = coordinate[0]
   y = coordinate[1]
   x = (x / 100) * 256
   y = (y / 100) * 256
   pos_list.append((x, y))

  print(pos_list)
  prediction.append((some_result.cpu().detach().numpy(), pos_list))
 return prediction

def get_densest_numpy_patches(pred):
  image, pos_list = pred
  f2 = plt.figure(30)
  plt.imshow(image, interpolation='nearest')
  plt.show()
  print("original")
  radius = 10
  maximum_value_list = []
  # get maximum value coordinates
  a = np.argmax(image)
  l = image.shape[0]
  c = a % l
  r = int(a / l)
  # make threshold the first highest value that could be found
  threshold = image[r, c] * 0.5

  # for every pixel that is at least as intens as the first
  # Annahme: Alle Säcke haben am Ende wenigstens einen Pixel mit der höchsten intensität. Ansonsten einfach eine Range von 5 % einrichten threshold * 0.95
  # while image[r,c] >= threshold:
  NUMBER_OF_HEX = 8
  for i in range(NUMBER_OF_HEX):
   if (r < 5 or c < 5):  # rand ausparen
    passes = False
    while (passes == False):
     print("suchen")
     image = cv2.circle(image, (c, r), 2, 0, -1)
     a = np.argmax(image)
     l = image.shape[0]
     c = a % l
     r = int(a / l)
     if (r > 5 and c > 5):
      if (image[r, c] < threshold):
       return maximum_value_list
   # Alle pixel darum auf 0 setzten damit argmax neuen höchsten finden kann
   image = cv2.circle(image, (c, r), radius, 0, -1)
   f1 = plt.figure(i)
   plt.imshow(image, interpolation='nearest')
   plt.show()
   # Höhepunkt hinzufügen
   maximum_value_list.append([c, r])
   a = np.argmax(image)
   c = a % 256
   r = int(a / 256)
   if (image[r, c] < threshold):
    return maximum_value_list

  return maximum_value_list

def save_as_json(list_max_value_ordered):
 print(list_max_value_ordered)
 json_predicted = {}
 json_predicted['rois'] = []
 frame = 0
 for i in list_max_value_ordered:
  id = 0
  for j in i:
   json_predicted['rois'].append({
    'z': frame,
    'id': id,
    'pos': j
   })
   id += 1
  frame += 1
 print(json_predicted)
 return json_predicted

def get_score(json_predicted):
 frameHeight = 256
 frameWidth = 256
 x = 0
 for i in json_predicted["rois"]:
  pos = i['pos']
  pos_x = np.round((pos[0] / 256) * frameHeight, 2)
  pos_y = np.round((pos[1] / 256) * frameWidth, 2)
  i['pos'] = [pos_x, pos_y]
 """
 for i in label_json["rois"]:
   pos = i['pos']
   pos_x = np.round(pos[0],2)
   pos_y = np.round(pos[1],2)
   i['pos'] = [pos_x,pos_y]
 """
 print(json_predicted['rois'])
 x = 0
 distmean = 0
 distsum = 0
 dist5 = 0
 dist10 = 0
 dist20 = 0
 dist30 = 0
 distdrüber = 0
 for i in json_predicted['rois']:
  dist = 0
  predx = i['pos'][0]
  predy = i['pos'][1]
  labelx = whatever_dict[x][0]
  labely = whatever_dict[x][1]
  dist = np.sqrt((labelx - predx) ** 2 + (labely - predy) ** 2)
  if (dist <= 5):
   dist5 += 1
  elif (dist <= 10):
   dist10 += 1
  elif (dist <= 20):
   dist20 += 1
  elif (dist <= 30):
   dist30 += 1
  elif (dist > 30):
   distdrüber += 1
  distsum += dist
  distmean += dist
  distmean = distmean / 2
  x += 1
 print("distmean: ", distmean)
 print("distsum: ", distsum)
 print("dist5: ", dist5 / x, "%")
 print("dist10: ", dist10 / x, "%")
 print("dist20: ", dist20 / x, "%")
 print("dist30: ", dist30 / x, "%")
 print("distdrüber: ", distdrüber / x, "%")


if __name__ == '__main__':
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 #newReader = DataReader()
 #model = Model.start_train()
 model = torch.load('C:/Users/Florian/Desktop/Job/Cornhole/Cornhole/model_cornhole.pth', map_location=torch.device('cpu'))

 prediction = get_test_images()
 list_max_value_unordered = []
 for i in prediction:
  list = get_densest_numpy_patches(i)
  print(list)
  list_max_value_unordered.append(list)

 json = save_as_json(list_max_value_unordered)
 get_score(json)