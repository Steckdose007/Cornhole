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
        self.directory_images = "C:/Users/Florian/Desktop/Job/cc_re/"
        # video list in which first the path to frame is stored. Then the frame number and then the position of the bag
        self.images_list = []
        # call funktion to store each frame
        self.read_images()
        x_train, x_test = train_test_split(self.images_list, test_size=0.2)
        print(len(self.images_list), len(x_train), len(x_test))
        pickle.dump(x_train, open("C:/Users/Florian/Desktop/Job/Cornhole/frames_train.p", "wb"))
        pickle.dump(x_test, open("C:/Users/Florian/Desktop/Job/Cornhole/frames_test.p", "wb"))

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
                traco_path = "C:/Users/Florian/Desktop/Job/Cornhole/annotat.json"
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
                path_to_frame = "C:/Users/Florian/Desktop/Job/cc_re/" + filename[:-4]
                # store frame
                pickle.dump(img, open(path_to_frame + ".p", "wb"))
                # store path to frame in video list
                self.images_list.append((path_to_frame + ".p", pos))
                os.remove("C:/Users/Florian/Desktop/Job/cc_re/" + filename)
                print(filename, pos)

                # for curr_frame in range(frameCount):
                #    self.video_list.append((self.directory_images+filename[:-3]+"p",curr_frame,whatever_dict[curr_frame]))
                continue
            else:
                continue
# To create a dataset class, I need a path-file that contains all of the train paths and another one for the test-paths
# This is necessary to allow multiple dataset instances to work in parallel


class TracoDataset(Dataset):
    # load the pickl lists
    def __init__(self, mode, transform=None):
        if mode == 'train':
            self.samples = pickle.load(open("C:/Users/Florian/Desktop/Job/Cornhole/frames_train.p", "rb"))
        elif mode == 'test':
            self.samples = pickle.load(open("C:/Users/Florian/Desktop/Job/Cornhole/frames_test.p", "rb"))
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
            RADIUS = 3
            base = cv2.circle(base, (int(pos_x), int(pos_y)), RADIUS, 100, -1)

        im_rgb = self.transform(im_rgb)
        base = self.transform(base)
        print(type(im_rgb), type(base))
        return im_rgb, base


class loss():
    #
    def dice_loss(pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

        return loss.mean()




def convrelu(in_channels, out_channels, kernel, padding):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        torch.nn.ReLU(inplace=True),
    )




def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.9):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


class ResNetUNet(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        # self.hidden_size1 = 64
        # self.hidden_size2
        # self.hidden_size3
        # self.hidden_size4

        self.layer0 = torch.nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = torch.nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = torch.nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)  # skip connections
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_densest_numpy_patches(pred):
  image,pos_list = pred
  f2 = plt.figure(30)
  plt.imshow(image, interpolation='nearest')
  plt.show()
  print("original")
  radius = 10
  maximum_value_list = []
  #get maximum value coordinates
  a = np.argmax(image)
  l=image.shape[0]
  c = a%l
  r = int(a/l)
  #make threshold the first highest value that could be found
  threshold = image[r,c]*0.5

  #for every pixel that is at least as intens as the first
  #Annahme: Alle Säcke haben am Ende wenigstens einen Pixel mit der höchsten intensität. Ansonsten einfach eine Range von 5 % einrichten threshold * 0.95
  #while image[r,c] >= threshold:
  NUMBER_OF_HEX = 8
  for i in range(NUMBER_OF_HEX):
    if( r < 5 or c < 5 ): #rand ausparen
      passes = False
      while(passes == False):
        print("suchen")
        image = cv2.circle(image, (c,r), 2,0, -1)
        a = np.argmax(image)
        l=image.shape[0]
        c = a%l
        r = int(a/l)
        if(r >5 and c>5):
          if(image[r,c]<threshold):
            return maximum_value_list
    #Alle pixel darum auf 0 setzten damit argmax neuen höchsten finden kann
    image = cv2.circle(image, (c,r), radius,0, -1)
    f1 = plt.figure(i)
    plt.imshow(image, interpolation='nearest')
    plt.show()
    #Höhepunkt hinzufügen
    maximum_value_list.append([c,r])
    a = np.argmax(image)
    c = a%256
    r = int(a/256)
    if(image[r,c]<threshold):
      return maximum_value_list

  return maximum_value_list

def get_test_images():
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

if __name__ == "__main__":
    #newReader = DataReader()

    train_transform = transforms.Compose([transforms.ToTensor()])
    print(torch.cuda.is_available())
    some_dataset = TracoDataset(mode='train', transform=train_transform)
    img, target = some_dataset.__getitem__(5)
    fig, axs = plt.subplots(2)
    target = torch.squeeze(target)
    target = target.numpy()
    axs[0].imshow(target, interpolation='nearest')
    img = torch.squeeze(img)
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    axs[1].imshow(img, interpolation='nearest')
    plt.show()

    # make dataloder

    normalize = transforms.Normalize((0.1307,), (0.3081,))

    train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    # make test and train set
    train_set = TracoDataset(transform=train_transform, mode='train')
    val_set = TracoDataset(transform=train_transform, mode='test')

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 16
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    }

    NUM_EPOCHS = 25
    N_CLASS = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNetUNet(N_CLASS).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    model = train_model(model, optimizer_ft, None, NUM_EPOCHS)


    prediction = get_test_images()
    list_max_value_unordered = []
    for i in prediction:
        list = get_densest_numpy_patches(i)
        print(list)
        list_max_value_unordered.append(list)
