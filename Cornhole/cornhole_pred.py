import RPi.GPIO as GPIO
import time
import telepot
from configparser import ConfigParser
import cv2
import torch 
import numpy as np
from torchvision import transforms, datasets, models
from torchvision.models import resnet18, ResNet18_Weights
import math
#from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

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


from collections import defaultdict
import torch.nn.functional as F


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
        #self.hidden_size1 = 64
        #self.hidden_size2
        #self.hidden_size3
        #self.hidden_size4

        self.layer0 = torch.nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = torch.nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
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
        x = torch.cat([x, layer3], dim=1)#skip connections
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
model = torch.load('/home/pi/model_cornhole.pth',map_location=torch.device('cpu'))
print("model geladen")
print("Programm Start")
#cam = cv2.VideoCapture(0)
print("Kamera offen")

config = ConfigParser()
config.read("config_cornhole.ini")
bot_id = config["BOT"]["bot_id"]
chat_id = int(config["BOT"]["channel_id"])

#bot = telepot.Bot(bot_id)
print("bot erstellt")

channel = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(channel,GPIO.IN)
i=0

def get_points(pos_list,img):
    teamorange=0
    teamblue=0
    for i in pos_list:
        x = int(i[0])
        y = int(i[1])
        rgb = img[x,y]
        print(x,y)
        print("rgb: ",rgb)
        r=rgb[0]
        g=rgb[1]
        b=rgb[2]
        #organgener sack
        if  0 <r< 255 and 0 <g< 178 and 0 <b< 35:
            teamorange +=1
        #blauer sack
        if 0 < r < 20 and 0 < g < 102 and 0 < b < 255:
            teamblue +=1
    return [teamorange,teamblue]

def prediction(img):
    global device
    global model
    model.eval()
    model = model.to(device)
    train_transform = transforms.Compose([transforms.ToTensor()])
    #cv2.resize(img, (256,256))
    print("image resized",type(img))
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_rgb = train_transform(im_rgb)
    img = torch.unsqueeze(im_rgb,0).to(device)
    some_result = model.forward(img)
    some_result = torch.squeeze(some_result,0)
    some_result = torch.squeeze(some_result,0)
    some_result = some_result.cpu().detach().numpy()
    print("Sacke estimated")
    return some_result

def get_densest_numpy_patches(pred):
  image = pred
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
            if (image[r, c] < threshold):
                return maximum_value_list
            passes = True
    #Alle pixel darum auf 0 setzten damit argmax neuen hochsten finden kann
    image = cv2.circle(image, (c,r), radius,0, -1)
    #Hohepunkt hinzufugen
    maximum_value_list.append([c,r])
    a = np.argmax(image)
    c = a%256
    r = int(a/256)
    if (image[r, c] < threshold):
        return maximum_value_list
  return maximum_value_list

#funktion to execute steps after movement
def do_sth():
    global i
    print("Movement")
    image = cv2.imread("/home/pi/IMG_3461.JPG",0)

    #ret, image = cam.read()
    #cv2.imwrite("/home/pi/Pictures/image_Rasp_"+str(i)+".jpg",image)
    if image is None:
        print("Unable to read image")
    else:
        print(type(image))#numpy.ndarray
        result_prediction = prediction(image)
        pos = get_densest_numpy_patches(result_prediction)
        #cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image[75,130],type(image),np.shape(image[pos[0]]),np.shape(image),np.shape(image[75,130]))
        points = get_points(pos,image)
        print("Points Orange: ",points[0],"Points Blue: ",points[1])

    i +=1
    #bot.sendMessage(chat_id, "Movement -> Picture taken")
    time.sleep(1)

print("Lasset die Säcke kommen.......")	
def callback(channel):
    if GPIO.input(channel):
        do_sth()
		
GPIO.add_event_detect(channel, GPIO.BOTH, bouncetime=400)
GPIO.add_event_callback(channel, callback)


while True:
    time.sleep(1)
