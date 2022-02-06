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
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

INPUT_DIR = Path.cwd() / "jaffedbase prepaired"
OUTPUT_DIR = Path.cwd() / "jaffedbase augmented"
# create output dir if not exists
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
# loop over all directories in the input directory
for dir in INPUT_DIR.iterdir():
    # loop over all files in the directory
    for file in dir.iterdir():
        # load the image and convert it to grayscale
        image = Image.open(file)
        image = image.convert("LA")
        imgmat = np.array(list(image.getdata(band=0)), float)
        imgmat.shape = (image.size[1], image.size[0])
        imgmat = np.matrix(imgmat)
        U, S, V = np.linalg.svd(imgmat)
        # reconstitute the image
        imgmat = np.dot(U, np.dot(np.diag(S), V))
        # create output path in the output directory
        outpath = OUTPUT_DIR / dir.name
        outpath.mkdir(exist_ok=True)
        # save image in output path + file + "_svd" dot original file ending
        plt.imsave(outpath / file.name.replace(file.suffix, "_svd" + file.suffix), imgmat, cmap="gray")
