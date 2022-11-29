import RPi.GPIO as GPIO
import time
import telepot
from configparser import ConfigParser
import cv2
import torch 
import numpy as np
from torchvision import transforms, datasets, models

print("model geladen")
print("Programm Start")
cam = cv2.VideoCapture(0)
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

#funktion to execute steps after movement
def do_sth():
    global i
    print("Movement")
    ret, image = cam.read()
    cv2.resize(image, (256, 256))
    cv2.imwrite("/home/pi/Pictures/image_Rasp_"+str(i)+".jpg",image)
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





