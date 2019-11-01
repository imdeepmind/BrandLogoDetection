import cv2
import os
from tqdm import tqdm
import random

def split(arr, split_ratio):
    random.shuffle(arr)
    
    n = len(arr)
    
    x,y,z = split_ratio
    
    train_size = int(n * x)
    validation_size = int(n*y)
    test_size = int(n*z)
    
    train = arr[:train_size]
    validation = arr[train_size:train_size+validation_size]
    test = arr[train_size+validation_size:train_size+validation_size+test_size]
    
    return train, validation, test

def process_image(path, type):
    try:
        if not os.path.exists('datasets/processd/' + type + '/' + path.split('/')[2]):
            os.makedirs('datasets/processd/' + type + '/' + path.split('/')[2])
            
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite('datasets/processd/' + type + '/' + path.split('/')[2] + '/' + path.split('/')[3], img)
    except Exception as ex:
        print(ex)

path = 'datasets/images'
classes = os.listdir(path)

split_ratio = (0.8, 0.1, 0.1)

train_data = []
validation_data = []
test_data = []

for cls in classes:
    images = os.listdir(path + '/' + cls)
    train, validation, test = split(images, split_ratio)
    
    [ train_data.append(path + '/' + cls + '/' + t) for t in train ]
    [ validation_data.append(path + '/' + cls + '/' + t) for t in validation ]
    [ test_data.append(path + '/' + cls + '/' + t) for t in test ]

for img in tqdm(train_data):
    process_image(img, 'train')

for img in tqdm(validation_data):
    process_image(img, 'validation')
    
for img in tqdm(test_data):
    process_image(img, 'test')
