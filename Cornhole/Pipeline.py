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
from Model import ResNetUNet

#loads first i images from train data, estimates pos with model
# and stores output in prediction
#returns a list with all predictions and the acording labels
def get_test_images(i):
 from torchvision import transforms
 train_transform = transforms.Compose([transforms.ToTensor()])
 samples = pickle.load(open("C:/Users/Florian/Desktop/Job/Cornhole/frames_train.p", "rb"))
 prediction = []
 for i in range(i):
  i = i + 10
  img = pickle.load(open(samples[i][0], "rb"))
  #get labels to image
  pos_tupel = samples[i][1]
  # convert to RGB
  im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  im_rgb = train_transform(im_rgb)
  img = torch.unsqueeze(im_rgb, 0).to(device)
  #put image in model
  some_result = model(img)
  some_result = torch.squeeze(some_result, 0)
  some_result = torch.squeeze(some_result, 0)

  #plot image and prediction
  fig, axs = plt.subplots(2)
  axs[0].imshow(some_result.cpu().detach().numpy(), interpolation='nearest')
  im_rgb = torch.squeeze(im_rgb)
  im_rgb = im_rgb.numpy()
  im_rgb = np.transpose(im_rgb, (1, 2, 0))
  axs[1].imshow(im_rgb, interpolation='nearest')
  plt.show()

  # posliste auf 256 bild converten
  pos_list = []
  for coordinate in pos_tupel:
   x = coordinate[0]
   y = coordinate[1]
   x = (x / 100) * 256
   y = (y / 100) * 256
   pos_list.append((x, y))

  prediction.append((some_result.cpu().detach().numpy(), pos_list))
 return prediction

#gets prediction and searches for max values which are the pos
#returns postions and labels
def get_densest_numpy_patches(pred):
  image, pos_list = pred
  fig, axs = plt.subplots(9)
  axs[0].imshow(image, interpolation='nearest')
  #print("original")
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
      # terminate kreterium when there are no more data points which are at least threshold*highest intensity pixel
      #and are not colser to the edge than 5 pixel, because of edge effects
      if (image[r, c] < threshold):
       plt.show()
       return maximum_value_list, pos_list
   # Alle pixel darum auf 0 setzten damit argmax neuen höchsten finden kann
   image = cv2.circle(image, (c, r), radius, 0, -1)
   axs[i+1].imshow(image, interpolation='nearest')

   # Höhepunkt hinzufügen
   maximum_value_list.append([c, r])
   a = np.argmax(image)
   c = a % 256
   r = int(a / 256)
   #terminate kreterium when there are no more datapoints which are at least threshold*highest intensity pixel
   if (image[r, c] < threshold):
    plt.show()
    return maximum_value_list, pos_list
  plt.show()
  return maximum_value_list, pos_list

def save_as_json(list_max_value_ordered):
 #print(list_max_value_ordered)
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
 #print(json_predicted)
 return json_predicted

#calculates the shortest distance to any label in reach
def get_score(predicted,label_list,num_images):
 #range of score
 x = 0
 distmean = 0
 distsum = 0
 dist2 =0
 dist5 = 0
 dist10 = 0
 dist20 = 0
 dist30 = 0
 distdrüber = 0
 nichtalleerkannt=0
 #go over all images as defined in main
 for i in range(num_images):
  #get label and pred
  pos_list_label = label_list[i]
  pos_list_pred = predicted[i]
  #case if not all bags are detected
  if(len(pos_list_pred) != len(pos_list_label)):
   nichtalleerkannt+=1
  else:
   #calculate shortest distance for a pred for every label and save
   for pred in pos_list_pred:
    dist = 1000
    for label in pos_list_label:
     d = math.dist(pred,label)
     if(d<dist):
      dist = d
   if (dist <= 2):
    dist2 += 1
   elif (dist < 5):
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
   x += 1

 print("distmean: ", distmean/x)
 print("distsum: ", distsum)
 print("dist2: ", (dist2 / x)*100, "%")
 print("dist5: ", (dist5 / x)*100, "%")
 print("dist10: ", (dist10 / x)*100, "%")
 print("dist20: ", (dist20 / x)*100, "%")
 print("dist30: ", (dist30 / x)*100, "%")
 print("distdrüber: ", (distdrüber / x)*100, "%")
 print('nichterkannt: ', nichtalleerkannt)
 print('x: ',x)

def plot_pred_and_labelpoints(pred,label,num):
 fig, axs = plt.subplots(num)
 for i in range(num):
  # get label and pred
  pos_list_label = label[i]
  pos_list_pred = pred[i]
  base = np.zeros([256, 256], dtype=np.uint8)
  for elem in pos_list_label:
   # reduziert die pos auf 256,256 bild
   pos_x = (elem[0] / 100) * 256
   pos_y = (elem[1] / 100) * 256
   # print(pos_x,pos_y)
   RADIUS = 3
   base = cv2.circle(base, (int(pos_x), int(pos_y)), RADIUS, 100, -1)
  for elem in pos_list_label:
   # reduziert die pos auf 256,256 bild
   pos_x = (elem[0] / 100) * 256
   pos_y = (elem[1] / 100) * 256
   # print(pos_x,pos_y)
   RADIUS = 3
   base = cv2.circle(base, (int(pos_x), int(pos_y)), RADIUS,color = (0, 0, 255), thickness= -1)
  axs[i].imshow(base, interpolation='nearest')
  plt.show()



if __name__ == '__main__':
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 #newReader = DataReader()
 #model = Model.start_train()
 model = torch.load('model_cornhole.pth', map_location=torch.device('cpu'))
 print('model loaded')
 num_images_to_test = 1
 prediction = get_test_images(num_images_to_test)

 list_max_value_unordered = []
 label_list=[]

 #for every prediction and acording label get positions
 for i in prediction:
  list,label = get_densest_numpy_patches(i)
  list_max_value_unordered.append(list)
  label_list.append(label)
 json = save_as_json(list_max_value_unordered)

 #print(json)
 #plot_pred_and_labelpoints(list_max_value_unordered,label_list,num_images_to_test)
 get_score(list_max_value_unordered,label_list,num_images_to_test)