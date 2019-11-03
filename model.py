import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from tqdm import tqdm

class BrandClassifier:
    def loadData(self):
        transform_train = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                              transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        transform_val_test = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        training_data = datasets.ImageFolder('{}/{}'.format(self.data, 'train'), transform=transform_train)
        validation_data = datasets.ImageFolder('{}/{}'.format(self.data, 'validation'), transform=transform_val_test)
        testing_data = datasets.ImageFolder('{}/{}'.format(self.data, 'test'), transform=transform_val_test)
        
        self.training_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
        self.testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=32, shuffle=False)
    
    def makeModel(self):
        self.model = models.mobilenet_v2(pretrained=True)
        
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        n_inputs = self.model.classifier[1].in_features
        last_layer = nn.Linear(n_inputs, 32)
        self.model.classifier[1] = last_layer
        self.model.to(self.device)
        
        print(self.model)
    
    def trainModel(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        
        epochs = 5
        self.losses = []
        
        for e in range(epochs):
            for inputs, labels in tqdm((self.training_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                output = self.model(inputs)
                loss = criterion(output, labels)
                
                self.losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
    def __init__(self, path):
        self.data = path
        
        # Find the best device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Running the model in ', self.device.type)
        
        # Train the model
        # Test the model

clf = BrandClassifier('datasets/processed')
clf.loadData()
clf.makeModel()
clf.trainModel()