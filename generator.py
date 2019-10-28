import pandas as pd
import cv2
import numpy as np

class Generator:
    train_counter = 0
    validation_counter = 0
    test_counter = 0
    
    def __init__(self, batch_size=32, split=(0.8,0.1,0.1)):
        data = pd.read_csv('datasets/labels.csv')
        
        data = data.sample(frac=1)
        
        X = data['id'].values
        Y = pd.get_dummies(data['label']).values
        
        x,y,z = split
        
        length = X.shape[0]
        
        train_size = int(length * x)
        validation_size = int(length * y)
        test_size = int(length * z)
        
        self.train = [X[:train_size], Y[:train_size]]
        self.validation = [X[train_size:train_size+validation_size], Y[train_size:train_size+validation_size]]
        self.test = [X[train_size+validation_size:train_size+validation_size+test_size], Y[train_size+validation_size:train_size+validation_size+test_size]]
        
        self.batch_size = batch_size
    
    def train_generator(self):
        while True:
            imgs = self.train[0][self.train_counter:self.train_counter+self.batch_size]
            labels = self.train[1][self.train_counter:self.train_counter+self.batch_size]
            
            self.train_counter += 1
            
            images = []
            
            for img in imgs:
                logo = cv2.imread('datasets/processed/' + img, 1) / 255.0
                images.append(logo)
            
            images = np.array(images)
            labels = np.array(labels).reshape(self.batch_size, 32)
            
            assert images.shape == (self.batch_size, 224, 224, 3), 'Invalid image shape'
            assert labels.shape == (self.batch_size, 32), 'Invalid labels shape'
            
            yield images, labels[0]
    
    def validation_generator(self):
        while True:
            imgs = self.validation[0][self.validation_counter:self.validation_counter+self.batch_size]
            labels = self.validation[1][self.validation_counter:self.validation_counter+self.batch_size]
            
            self.validation_counter += 1
            
            images = []
            
            for img in imgs:
                logo = cv2.imread('datasets/processed/' + img, 1) / 255.0
                images.append(logo)
            
            images = np.array(images)
            labels = np.array(labels).reshape(self.batch_size, 32)
            
            assert images.shape == (self.batch_size, 224, 224, 3), 'Invalid image shape'
            assert labels.shape == (self.batch_size, 32), 'Invalid labels shape'
            
            yield images, labels[0]
    
    def test_generator(self):
        while True:
            imgs = self.test[0][self.test_counter:self.test_counter+self.batch_size]
            labels = self.test[1][self.test_counter:self.test_counter+self.batch_size]
            
            self.test_counter += 1
            
            images = []
            
            for img in imgs:
                logo = cv2.imread('datasets/processed/' + img, 1) / 255.0
                images.append(logo)
            
            images = np.array(images)
            labels = np.array(labels).reshape(self.batch_size, 32)
            
            assert images.shape == (self.batch_size, 224, 224, 3), 'Invalid image shape'
            assert labels.shape == (self.batch_size, 32), 'Invalid labels shape'
            
            yield images, labels[0]
    
gen = Generator(batch_size=2)
x, y = next(gen.train_generator())