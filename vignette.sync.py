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
import numpy as np
import cv2
from pathlib import Path
import imutils

# %%
INPUT_DIR = Path.cwd() / "jaffedbase prepaired"
OUTPUT_DIR = Path.cwd() / "jaffedbase augmented"
# create output dir if not exists
OUTPUT_DIR.mkdir(exist_ok=True)
# loop over the input directory looking for subdirectories
for dir in Path(INPUT_DIR).iterdir():
    if dir.is_dir():
        # loop over the images in each directory + augmentation
        for imagePath in dir.iterdir():
            # load the input image from disk
            image = cv2.imread(str(imagePath))
            # randomly flip the image horizontally
            if np.random.randint(2) == 0:
                image = cv2.flip(image, 1)
            # randomly flip the image vertically
            if np.random.randint(2) == 0:
                image = cv2.flip(image, 0)
            # randomly rotate the image by the specified angle
            angle = np.random.randint(-15, 15)
            try:
                image = imutils.rotate(image, angle)
            except:
                print("rotation failed", imagePath)
                continue
            # write the image to disk
            filename = imagePath.name
            p = Path(OUTPUT_DIR) / dir.name
            p.mkdir(exist_ok=True)
            cv2.imwrite(str(p / filename), image)
