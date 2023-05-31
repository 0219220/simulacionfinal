from skimage.io import imread, imshow
from skimage import color, filters
import csv
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

imgstyles=["Blade Runner 2049", "God of War Ascension", "Horikoshi", "Magritte", "Oceano","Perros", "Rembrandt", "StarWars", "USM"]
names =["Villeneuve_","GOWA ","","Magritte_", "Mar_", "Perros_", "Rembrandt_", "StarWars_", "USM_"]

def fextraction(image, name, imglocation):

    gray_image=color.rgb2gray(image)
    hsv_image = color.rgb2hsv(image)

    red_channel = image[:,:,0]
    green_channel = image[:,:,1]
    blue_channel = image[:,:,2]

    mean = np.mean(image)
    variance = np.mean([abs(p - mean)**2 for p in np.nditer(image)])
    contrast = np.sqrt(variance)
    size = image.shape[0]*image.shape[1]
    ###Feature 1

    bpnum = np.sum(blue_channel>20)
    rpnum = np.sum(red_channel>20)

    ##Feature2
    brightness = np.mean(gray_image) ##Brightness

    img_threshold = 0.5 ##Amount of white pixels
    wbinary_image = gray_image > img_threshold
    wpixels = np.count_nonzero(wbinary_image)
    whites = wpixels/ size

    bbinary_image = gray_image <= img_threshold ##Amount of black pixels
    bpixels = np.count_nonzero(bbinary_image)
    blacks = bpixels / size

    ##Feature3
    rpaverage=np.mean(red_channel) ###Saturation of red, green and blue channels
    gpaverage=np.mean(green_channel)
    bpaverage=np.mean(blue_channel)

    ##Feature4
    bwthresh=filters.threshold_otsu(gray_image) ###Extracting the white and black pixels from an image
    binary=image <= img_threshold
    bapixels= np.sum(binary == 0)
    wapixels= np.sum(binary == 1)

    ##Feature5
    bpnum = np.sum(blue_channel>20)
    bpavg = np.mean(blue_channel)
    saturationb = np.mean(hsv_image[:, :, 1])

    return [name, imglocation,mean, variance, contrast, bpnum, rpnum, brightness, whites, blacks, rpaverage, gpaverage, bpaverage, bapixels, wapixels, bpnum, bpavg, saturationb]

with open('features.csv', 'w', newline='') as csvfile:
    writefile = csv.writer(csvfile)
    writefile.writerow(['File','Name','imglocation', 'Mean', 'Variance', 'Contrast', 'B pixels amount', 'R pixels amount', 'Brightness', 'Whites', 'Blacks' 'R pixel average', 'G pixel average', 'B pixel average', 'White pixels', 'Black pixels', 'Blue Saturation'])

    for j in range(9):
        for i in range(1, 100):
            name = names[j]+str(i)+'.png'
            image = cv2.imread('./images/'+imgstyles[j]+'/'+name)
            features = fextraction(image, name, imgstyles[j])
            writefile.writerow([os.path.basename(str(i))]+ features)

print("Ready")