#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

setup(
   name='cornhole',
   version='0.0',
   author='Florian Gritsch',
   author_email='florian@gritsch.eu',
   packages=find_packages(),
   install_requires= ['numpy','posixpath','json', 'cv2','os','pickle','pathlib','sklearn','torch','torchvision','matplotlib','collections','copy','math','time'],
)