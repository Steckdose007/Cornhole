import os
import cv2
from PIL import Image
print("Start:")

#folder with original images
path = r'C:/Users/Florian/Desktop/Job/cc'
#folder with resized images
path2= r'C:/Users/Florian/Desktop/Job/cc_re'

#what to resize it
mean_width = 256
mean_height = 256

# Resizing of the images to give
# them same width and height
sum =0
for file in os.listdir(path):
    if file.endswith(".JPG"):
        im = Image.open(os.path.join(path, file))
        imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS)
        imResize.save(os.path.join(path2, file), 'JPEG', quality=95)  # setting quality

        sum += 1
    print(sum)