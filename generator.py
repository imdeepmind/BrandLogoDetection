import pandas as pd
import cv2
import numpy as np

class Generator:
    train_counter = 0
    validation_counter = 0
    test_counter = 0
    
    def __init__(self, batch_size=32, split=(0.8,0.1,0.1)):
        data = pd.read_csv('datasets/labels.csv')
        
        data = data.sample(frac=1).values
        
        x,y,z = split
        
        length = data.shape[0]
        
        train_size = int(length * x)
        validation_size = int(length * y)
        test_size = int(length * z)
        
        self.train = data[:train_size]
        self.validation = data[train_size:validation_size]
        self.test = data[train_size+validation_size:test_size]
        
        self.batch_size = batch_size
    
    def train_generator(self):
        while True:
            images = []
            labels = []
            
            imgs = self.train[self.train_counter:self.train_counter+self.batch_size]
            
            self.train_counter += 1
            
            for img, lbl in imgs:
                logo = cv2.imread('datasets/processed/' + img, 1) / 255.0
                images.append(logo)
                labels.append(lbl)
            
            images = np.array(images)
            labels = np.array(labels).reshape(self.batch_size, 1)
            
            assert images.shape == (self.batch_size, 224, 224, 3), 'Invalid image shape'
            assert labels.shape == (self.batch_size, 1), 'Invalid labels shape'
            
            yield images, labels
    
    def validation_generator(self):
        while True:
            images = []
            labels = []
            
            imgs = self.validation[self.validation_counter:self.validation_counter+self.batch_size]
            
            self.validationn_counter += 1
            
            for img, lbl in imgs:
                logo = cv2.imread('datasets/processed/' + img, 1) / 255.0
                images.append(logo)
                labels.append(lbl)
            
            images = np.array(images)
            labels = np.array(labels).reshape(self.batch_size, 1)
            
            assert images.shape == (self.batch_size, 224, 224, 3), 'Invalid image shape'
            assert labels.shape == (self.batch_size, 1), 'Invalid labels shape'
            
            yield images, labels
    
    
    def test_generator(self):
        while True:
            images = []
            labels = []
            
            imgs = self.test[self.test_counter:self.test_counter+self.batch_size]
            
            self.test_counter += 1
            
            for img, lbl in imgs:
                logo = cv2.imread('datasets/processed/' + img, 1) / 255.0
                images.append(logo)
                labels.append(lbl)
            
            images = np.array(images)
            labels = np.array(labels).reshape(self.batch_size, 1)
            
            assert images.shape == (self.batch_size, 224, 224, 3), 'Invalid image shape'
            assert labels.shape == (self.batch_size, 1), 'Invalid labels shape'
            
            yield images, labels
    

gen = Generator(batch_size=2)
x, y = next(gen.train_generator())