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
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import cv2
from pathlib import Path

INPUT_DIR = Path.cwd() / "jaffedbase prepaired"
OUTPUT_DIR = Path.cwd() / "jaffedbase augmented"
# create output dir if not exists
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
# loop over all directories in input dir
for dir in INPUT_DIR.iterdir():
    # loop over all files in dir
    for file in dir.iterdir():
        try:
            # read image
            img = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
            # splitting into channels
            b, g, r = cv2.split(img)
            df_blue = b/255
            df_green = g/255
            df_red = r/255
            pca_b = PCA(n_components=50)
            pca_b.fit(df_blue)
            trans_pca_b = pca_b.transform(df_blue)
            pca_g = PCA(n_components=50)
            pca_g.fit(df_green)
            trans_pca_g = pca_g.transform(df_green)
            pca_r = PCA(n_components=50)
            pca_r.fit(df_red)
            trans_pca_r = pca_r.transform(df_red)
            b_arr = pca_b.inverse_transform(trans_pca_b)
            g_arr = pca_g.inverse_transform(trans_pca_g)
            r_arr = pca_r.inverse_transform(trans_pca_r)
            img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
            # create OUTPUT_DIR + dir if doesn't exist
            output_path = OUTPUT_DIR / dir.name
            output_path.mkdir(exist_ok=True)
            lab_image = np.float32(img_reduced)
            # lab_image = cv2.cvtColor(lab_image, cv2.COLOR_RGB2HSV)
            # save image to output_path + file + '_pca' dot original extension
            cv2.imwrite(str(output_path / file.name.replace(file.suffix, '_pca' + file.suffix)), lab_image)
        except Exception as e:
            print(e, file)
            pass
