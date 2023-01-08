import posixpath
import json
import cv2
import os
import pickle
import pathlib
from numpy import array
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
from operator import add
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


# loads first i images from train data, estimates pos with model
# and stores output in prediction
# returns a list with all predictions and the acording labels
def get_test_images(i, Bool):
    from torchvision import transforms
    train_transform = transforms.Compose([transforms.ToTensor()])
    samples = pickle.load(open(
        "C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Job/Cornhole/frames_test.p",
        "rb"))
    print(len(samples))
    prediction = []
    color_images = []
    for i in range(i):
        i = i + 10
        img = pickle.load(open(samples[i][0], "rb"))
        # get labels to image
        pos_tupel = samples[i][1]
        # convert to RGB
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img5 = copy.deepcopy(im_rgb)
        img5 = np.asarray(img5)
        im_rgb = train_transform(im_rgb)
        img1 = torch.unsqueeze(im_rgb, 0).to(device)
        # put image in model
        img1 = img1.to(device)
        some_result = model(img1)
        some_result = torch.squeeze(some_result, 0)
        some_result = torch.squeeze(some_result, 0)
        if (Bool == True):
            # plot image and prediction
            fig, axs = plt.subplots(2)
            axs[0].imshow(some_result.cpu().detach().numpy())
            im_rgb = torch.squeeze(im_rgb)
            im_rgb = im_rgb.numpy()
            im_rgb = np.transpose(im_rgb, (1, 2, 0))
            axs[1].imshow(im_rgb)
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
        color_images.append(img5)
    return prediction, color_images


# gets prediction and searches for max values which are the pos
# returns postions and labels
def get_densest_numpy_patches(pred, Bool):
    image, pos_list = pred
    image_original = copy.deepcopy(image)
    if (Bool == True):
        fig, axs = plt.subplots(9)
        axs[0].imshow(image)
    # print("original")
    radius = 10
    maximum_value_list = []
    # get maximum value coordinates
    a = np.argmax(image)
    l = image.shape[0]
    c = a % l
    r = int(a / l)
    # make threshold the first highest value that could be found
    threshold = image[r, c] * 0.55

    # for every pixel that is at least as intens as the first
    # Annahme: Alle Säcke haben am Ende wenigstens einen Pixel mit der höchsten intensität. Ansonsten einfach eine Range von 5 % einrichten threshold * 0.95
    # while image[r,c] >= threshold:
    NUMBER_OF_HEX = 8
    for i in range(NUMBER_OF_HEX):
        if (r < 5 or c < 5):  # rand ausparen
            passes = False
            while (passes == False):
                # print("suchen")
                image = cv2.circle(image, (c, r), 2, 0, -1)
                a = np.argmax(image)
                l = image.shape[0]
                c = a % l
                r = int(a / l)
                if (r > 5 and c > 5):
                    # terminate kreterium when there are no more data points which are at least threshold*highest intensity pixel
                    # and are not colser to the edge than 5 pixel, because of edge effects
                    if (image[r, c] < threshold):
                        if (Bool == True):
                            plt.show()
                        return maximum_value_list, pos_list, image_original
        # Alle pixel darum auf 0 setzten damit argmax neuen höchsten finden kann
        image = cv2.circle(image, (c, r), radius, 0, -1)
        if (Bool == True):
            axs[i + 1].imshow(image)

        # Höhepunkt hinzufügen
        maximum_value_list.append([c, r])
        a = np.argmax(image)
        c = a % 256
        r = int(a / 256)
        # terminate kreterium when there are no more datapoints which are at least threshold*highest intensity pixel
        if (image[r, c] < threshold):
            return maximum_value_list, pos_list, image_original
    return maximum_value_list, pos_list, image_original


def save_as_json(list_max_value_ordered):
    # print(list_max_value_ordered)
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
    # print(json_predicted)
    return json_predicted


# calculates the shortest distance to any label in reach
def get_score(predicted, label_list, num_images):
    # range of score
    print("Score")
    x = 0
    distmean = 0
    distsum = 0
    dist2 = 0
    dist5 = 0
    dist10 = 0
    dist20 = 0
    dist30 = 0
    distdrüber = 0
    nichtalleerkannt = 0
    # go over all images as defined in main
    for i in range(num_images):
        # get label and pred
        pos_list_label = label_list[i]
        pos_list_pred = predicted[i]
        # case if not all bags are detected
        if (len(pos_list_pred) != len(pos_list_label)):
            nichtalleerkannt += (len(pos_list_label) - len(pos_list_pred))
        else:
            # calculate shortest distance for a pred for every label and save
            for pred in pos_list_pred:
                dist = 1000
                for label in pos_list_label:
                    d = math.dist(pred, label)
                    if (d < dist):
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

    print("distmean: ", distmean / x)
    print("distsum: ", distsum)
    print("dist2: ", (dist2 / x) * 100, "%")
    print("dist5: ", (dist5 / x) * 100, "%")
    print("dist10: ", (dist10 / x) * 100, "%")
    print("dist20: ", (dist20 / x) * 100, "%")
    print("dist30: ", (dist30 / x) * 100, "%")
    print("distdrüber: ", (distdrüber / x) * 100, "%")
    print('nichterkannteSacke: ', nichtalleerkannt)
    print('x: ', x)


# plottet die label auf das predictete image des models
# und die predicted coordinates auf das color_image
def plot_pred_and_labelpoints(pred, label, img_pred, img_color):
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_pred)
    axs[1].imshow(img_color)
    for elem in label:
        pos_x = elem[0]
        pos_y = elem[1]
        axs[0].scatter(pos_x, pos_y, color="darkorange")
    for elem in pred:
        pos_x = elem[0]
        pos_y = elem[1]
        axs[1].scatter(pos_x, pos_y, color="navy")
    plt.show()


def get_color_points(pos_list, img):
    teamorange = 0
    teamblue = 0
    #print("get color points")
    #print(pos_list)
    for i in pos_list:
        y = int(i[0])
        x = int(i[1])
        #print(img[x + 1, y], img[x + 1, y - 1], img[x + 1, y + 1], img[x - 1, y - 1], img[x - 1, y + 1], img[x - 1, y], img[x, y + 1], img[x, y - 1], img[x, y])
        l= [img[x + 1, y], img[x + 1, y - 1], img[x + 1, y + 1], img[x - 1, y - 1], img[x - 1, y + 1], img[x - 1, y], img[x, y + 1], img[x, y - 1], img[x, y]]
        rgb_sum = [0,0,0]
        for j in l:
            rgb_sum[0]+=j[0]
            rgb_sum[1]+=j[1]
            rgb_sum[2]+=j[2]
        rgb_sum[0] = rgb_sum[0]/9
        rgb_sum[1] = rgb_sum[1]/9
        rgb_sum[2] = rgb_sum[2]/9
        #print("rgb_sum :", rgb_sum)
        rgb = rgb_sum
        #print(x, y)
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        diff = b-r
        if(diff <0):
            teamorange += 1
            #print("orange")
        if(diff>=0):
            teamblue += 1
            #print("blue")
    return [teamorange, teamblue]

def evaluate_Punktevergabe(points_from_predicted,points_from_label):
    # wenn nicht gleich vieöe Sacke der verschiedenen Farben entdeck wurden
    #red
    if (points_from_predicted[0] == points_from_label[0]):
        Score_same_points[0][0] += 1
    else:
        Score_same_points[0][1] += abs(points_from_predicted[0] - points_from_label[0])
    if(points_from_predicted[1] == points_from_label[1]):
        Score_same_points[1][0] += 1
    else:
        Score_same_points[1][1] += abs(points_from_predicted[1] - points_from_label[1])


    #print("Orange_pred: ", points_from_predicted[0], "Blau_pred: ", points_from_predicted[1])
    #print("Orange_label: ", points_from_label[0], "Blau_label: ", points_from_label[1])



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # newReader = DataReader()
    # model = Model.start_train(15,16)
    model = torch.load('model_cornhole5.pth', map_location=torch.device('cpu'))
    if torch.cuda.is_available():
        model.cuda()
    print('model loaded')
    num_images_to_test = 200
    show_images = False
    prediction, color_images = get_test_images(num_images_to_test, show_images)

    list_max_value_unordered = []
    label_list = []
    Score_same_points=[[0,0],[0,0]]

    # for every prediction and acording label get positions
    j = 0
    for i in prediction:
        #get the coordinates of the sacken
        predicted_coordinates_list, label, image_pred = get_densest_numpy_patches(i, False)
        if (show_images == True):
            plt.show()
            plot_pred_and_labelpoints(predicted_coordinates_list, label, image_pred, color_images[j])

        #points hat erst orange dann blaue sacke
        points_from_predicted = get_color_points(predicted_coordinates_list, color_images[j])
        points_from_label = get_color_points(label, color_images[j])

        evaluate_Punktevergabe(points_from_predicted,points_from_label)

        list_max_value_unordered.append(predicted_coordinates_list)
        label_list.append(label)
        j += 1
    json = save_as_json(list_max_value_unordered)

    # print(json)
    print(Score_same_points[0][0],  " = anzahl der orangen gleich")
    print(Score_same_points[0][1], " = anzahl der orangen nicht erkannten")
    print(Score_same_points[1][0],  " = anzahl der blauen gleich")
    print(Score_same_points[1][1], " = anzahl der blauen nicht erkannten")
    get_score(list_max_value_unordered, label_list, num_images_to_test)
