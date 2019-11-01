import torch
from torchvision import datasets, transforms

class BrandClassifier:
    def __init__(self, data):
        self.data = data
    
    def dataLoader(self):
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
        
        training_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
        testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=32, shuffle=False)