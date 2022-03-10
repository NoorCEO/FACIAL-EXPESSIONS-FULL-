# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import cv2

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(266, 266)),
    transforms.RandomCrop(size=(265, 265)),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0, 0, 0), (1, 1, 1))
])

data_dir = os.getcwd() + '/jaffedbase prepaired/'
output_dir = os.getcwd() + '/jaffedbase augmented/'

# %%
# loop over folders in `data_dir`
for folder in os.listdir(data_dir):
    # loop over files in each folder
    for file in os.listdir(data_dir + folder):
        # load image
        img = cv2.imread(data_dir + folder + '/' + file)
        # apply transforms
        img = my_transforms(img)
        # create output directory if it doesn't exist
        if not os.path.exists(output_dir + folder):
            os.makedirs(output_dir + folder)
        # get filename without extension using `os.path.splitext`
        filename, extension = os.path.splitext(file)
        # save image
        save_image(img, output_dir + folder + '/transforms_' + filename + '.png')

