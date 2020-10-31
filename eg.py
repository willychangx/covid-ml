import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

import os

# TODO: Construct your data in the following baseline structure: 1) ./Dataset/Train/image/, 2) ./Dataset/Train/label, 3) ./Dataset/Test/image, and 4) ./Dataset/Test/label
class DataSet(Dataset):
    def __init__(self, root):
        
        currDir = os.getcwd()

        folderNames = root.split('/')
        currPath = currDir
        for name in folderNames:
            if name != '.' and name != '':
                try:
                    os.chdir(name)
                except:
                    os.mkdir(name)
                    os.chdir(name)

        os.mkdir('image')
        os.mkdir('label')

        os.chdir(currDir)
        
        self.ROOT = root
        self.images = read_images(root + "/image")
        self.labels = read_labels(root + "/label")

    def __len__(self):
        # Return number of points in the dataset

        return len(self.images)

    def __getitem__(self, idx):
        # Here we have to return the item requested by `idx`. The PyTorch DataLoader class will use this method to make an iterable for training/validation loop.

        img = images[idx]
        label = labels[idx]

        return img, label

# Load the dataset and train and test splits
print("Loading datasets...")

# Data path
DATA_train_path = Dataset('./Dataset/Train')
DATA_test_path = Dataset('./Dataset/Test')

# Data normalization
MyTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Convert image to grayscale
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
    transforms.Normalize([0.0], [0.1]) # TODO: Normalize to zero mean and unit variance with appropriate parameters
])

DATA_train = datasets.ImageFolder(root=DATA_train_path, transform=MyTransform)
DATA_test = datasets.ImageFolder(root=DATA_test_path, transform=MyTransform)

print("Done!")

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(Data_train, batch_size=??, shuffle=True)
testloader = DataLoader(Data_test, batch_size=??, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: [Transfer learning with pre-trained ResNet-50] 1) Define how many first layers of convolutoinal neural network (CNN) feature extractor in ResNet-50 to be "frozen" and 2) design your own fully-connected network (FCN) classifier.
        # 1) You will only refine last several layers of CNN feature extractor in ResNet-50 that mainly relate to high-level vision task. Determine how many first layers of ResNet-50 should be frozen to achieve best performances. Commented codes below will help you understand the architecture, i.e., "children", of ResNet-50.
        # 2) Design your own FCN classifier. Here I provide a sample of two-layer FCN.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, ReLU, Dropout, MaxPool2d, AvgPool2d
        # If you have many layers, consider using nn.Sequential() to simplify your code
        
        # Load pretrained ResNet-50
        self.model_resnet = models.resnet50(pretrained=True)
        
        # The code below can show children of ResNet-50
        #child_counter = 0
        #for child in model.children():
        #    print(" child", child_counter, "is -")
        #    print(child)
        #    child_counter += 1
        
        # TODO: Determine how many first layers of ResNet-50 to freeze
        child_counter = 0
        for child in model_resnet.children():
            if child_counter < ??:
                for param in child.parameters():
                    param.requires_grad = False
            elif child_counter == ??:
                children_of_child_counter = 0
                for children_of_child in child.children():
                    if children_of_child_counter < ???:
                        for param in children_of_child.parameters():
                            param.requires_grad = False
                    else:
                    children_of_child_counter += 1
            else:
                print("child ",child_counter," was not frozen")
            child_counter += 1
        
        # Set ResNet-50's FCN as an identity mapping
        num_fc_in = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        
        # TODO: Design your own FCN
        self.fc1 = nn.Linear(num_fc_in, ?, bias = ??) # from input of size num_fc_in to output of size ?
        self.fc2 = nn.Linear(?, 3, bias = ??) # from hidden layer to 3 class scores

    def forward(self,x):
        # TODO: Design your own network, implement forward pass here
        
        relu = nn.ReLU() # No need to define self.relu because it contains no parameters
        
        with torch.no_grad():
            features = self.model_resnet(x)
            
        x = self.fc1(features) # Activation are flattened before being passed to the fully connected layers
        x = relu(x)
        x = self.fc2(x)
        
        # The loss layer will be applied outside Network class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
model = Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer (note: CrossEntropyLoss already includes LogSoftMax())
# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=??, weight_decay=???) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength (default: lr=1e-2, weight_decay=1e-4)
num_epochs = ?? # TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = num_epochs): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc
    
train(model, trainloader, num_epochs)
print("Evaluate on test set")
evaluate(model, testloader)