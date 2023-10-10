""" Generate/Use NN to 'predict' Rs"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from IPython import embed

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

class SimpleNet(nn.Module):
    def __init__(self, ninput:int, noutput:int, 
                 nhidden:int):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(ninput, nhidden)
        self.fc2 = nn.Linear(nhidden, noutput)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x

def preprocess_data(data):

    # Normalize
    data = (data - data.mean(axis=0))/data.std(axis=0)

    return data.astype(np.float32)

def perform_training(model, dataset, ishape:int, train_kwargs, lr):

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset, **train_kwargs)

    epochs=100
    for epoch in range(epochs):
        loss = 0
        for batch_features, targets in train_loader:

            # load it to the active device
            batch_features = batch_features.view(-1, ishape).to(device)

            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training loss
            train_loss = criterion(outputs, targets)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

if __name__ == '__main__':

    # ##############################
    # Quick NN on L23

    # Load up data
    l23_path = os.path.join(os.getenv('OS_COLOR'),
                            'data', 'Loisel2023')
    outfile = os.path.join(l23_path, 'pca_ab_33_Rrs.npz')

    d = np.load(outfile)
    nparam = d['a'].shape[1]+d['b'].shape[1]
    ab = np.zeros((d['a'].shape[0], nparam))
    ab[:,0:d['a'].shape[1]] = d['a']
    ab[:,d['a'].shape[1]:] = d['b']

    target = d['Rs']

    # Preprocess
    pre_ab = preprocess_data(ab)
    pre_targ = preprocess_data(target)

    # Dataset
    dataset = MyDataset(pre_ab, pre_targ)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNet(nparam, target.shape[1], 128).to(device)
    train_kwargs = {'batch_size': 32}

    lr = 1e-3
    perform_training(model, dataset, nparam, 
                     train_kwargs, lr)

    # Fuss
    embed(header='126 of nn.py')