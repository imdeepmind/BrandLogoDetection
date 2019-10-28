"""
Created on Mon Oct 28 16:12:08 2019

@author: Abhishek Chatterjee (imdeepmind)

@description: A file for resizing all the images and storing it in proper way
"""

import cv2
import os
from tqdm import tqdm
import pandas as pd

# List of all classes
classes = os.listdir('datasets/images')

# List for storing the dataset
images = []
labels = []

# Making a dir for storing all processed images
if not os.path.exists('datasets/processed'):
    os.makedirs('datasets/processed')

# Going through all the images
for cls in classes:
    print('Processing {} logos'.format(cls))
    
    path = 'datasets/images/' + cls
    
    # List of all images for a specific class
    imgs = os.listdir(path)
    
    # Going through all the images of a specific class
    for index, img in tqdm(enumerate(imgs)):
        try:
            # Reading a specific image
            logo = cv2.imread(path + '/' + img, 1)
            
            # Resizing the image
            logo = cv2.resize(logo, (224,224))
            
            # Saving into specific dir
            cv2.imwrite('datasets/processed/' + str(index) + '_' + cls + '.jpg', logo)
            
            # Storing the info for the image
            images.append(str(index) + '_' + cls + '.jpg')
            labels.append(cls)
        except Exception as ex:
            print(str(ex))

data = pd.DataFrame(columns=['id', 'label'])

data['id'] = images
data['label'] = labels

data = data.sample(frac=1)

data.to_csv('datasets/labels.csv', index=False)