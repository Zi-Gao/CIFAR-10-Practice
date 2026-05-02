import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
import os
import torch.nn.functional as F
import requests
from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sn
import pandas as pd
torch.manual_seed(1) # Set manual seed
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import tqdm

torch.set_default_dtype(torch.float32)

def main():

    # Define GPU device - We will run our model on GPU
    mps_device = torch.device("mps")

    IMAGE_SIZE = 224 # We need to resize the images given resnet takes input of image size >= 224

    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    # These values are mostly used by researchers as found to very useful in fast convergence


    # https://pytorch.org/vision/stable/transforms.html
    # We can try various transformation for good generalization of model
    composed_train = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize the image in a 32X32 shape
                                        transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
                                        transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
                                        transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
                                                                contrast = 0.1, 
                                                                saturation = 0.1), 
                                        transforms.RandomAdjustSharpness(sharpness_factor = 2,
                                                                        p = 0.1), # Randomly adjust sharpness
                                        transforms.ToTensor(),   # Converting image to tensor
                                        transforms.Normalize(mean, std), # Normalizing with standard mean and standard deviation
                                        transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)])


    composed_test = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])


    # Load the data and transform the dataset
    train_dataset =  dsets.CIFAR10(root='./data/py', 
                                train=True, 
                                download=True, 
                                transform = composed_train)
    validation_dataset = dsets.CIFAR10(root='./data/py', 
                                    train=False, 
                                    download=True, 
                                    transform = composed_test)

    # Create train and validation batch for training
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100,num_workers=2,persistent_workers=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100,num_workers=2,persistent_workers=True)

    criterion = nn.CrossEntropyLoss()

    def train_model(model, train_loader, validation_loader, optimizer, scheduler, n_epochs = 20):
        
        # Global variable
        N_test = len(validation_dataset)
        accuracy_list = []
        train_loss_list = []
        model = model.to(mps_device)
        train_cost_list = []
        val_cost_list = []
        
        for epoch in tqdm.trange(n_epochs):
            train_COST = 0
            for x,y in tqdm.tqdm(train_loader):
                x = x.to(mps_device)
                y = y.to(mps_device)
                model.train()
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z,y)
                loss.backward()
                optimizer.step()
                train_COST+=loss.item()
                
            train_COST = train_COST/len(train_loader)
            train_cost_list.append(train_COST)
            correct = 0
            
            # Perform the prediction on the validation data
            val_COST = 0
            for x_test, y_test in tqdm.tqdm(validation_loader):
                model.eval()
                x_test = x_test.to(mps_device)
                y_test = y_test.to(mps_device)
                z = model(x_test)
                val_loss = criterion(z, y_test)
                _, yhat = torch.max(z.data, 1)
                correct += (yhat==y_test).sum().item()
                val_COST+=val_loss.item()
            
            val_COST = val_COST/ len(validation_loader)
            val_cost_list.append(val_COST)
                
            accuracy = correct / N_test
            accuracy_list.append(accuracy)

            scheduler.step(accuracy)
            
            print("--> Epoch Number : {}".format(epoch + 1),
                " | Training Loss : {}".format(round(train_COST,4)),
                " | Validation Loss : {}".format(round(val_COST,4)),
                " | Validation Accuracy : {}%".format(round(accuracy * 100, 2)))
            
        return accuracy_list, train_cost_list, val_cost_list


    def resnet_34():
        # Define the resnet model
        resnet = torchvision.models.resnet34(pretrained=True)
        
        # Update the fully connected layer of resnet with our current target of 10 desired outputs
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
        
        # Initialize with xavier uniform
        torch.nn.init.xavier_uniform_(resnet.fc.weight)
        return resnet

    model_mmtv5 = resnet_34()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model_mmtv5.parameters(), 
                                lr = learning_rate, 
                                momentum = 0.2)

    # Define the Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # Train the model
    accuracy_list_normalv5, train_cost_listv5, val_cost_listv5=train_model(model=model_mmtv5, 
                                                                        n_epochs=25, 
                                                                        train_loader=train_loader, 
                                                                        validation_loader=validation_loader, 
                                                                        optimizer=optimizer, 
                                                                        scheduler = scheduler)
    
if __name__=="__main__": main()