""" Generate/Use NN to 'predict' Rs"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)

class SimpleNet(nn.Module):
    def __init__(self, ninput:int, noutput:int, nhidden:int):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(ninput, nhidden)
        self.fc2 = nn.Linear(nhidden, noutput)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x


def perform_training(model, dataset, ishape:int, train_kwargs, lr):

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(dataset,
                                               **train_kwargs)

    epochs=10
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, ishape).to(device)
            
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
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

    # Dataset
    l23_path = os.path.join(os.getenv('OS_COLOR'),
                            'data', 'Loisel2023')
    outfile = os.path.join(l23_path, 'pca_ab_33_Rrs.npz')

    d = np.load(outfile)
    nparam = d['a'].shape[1]+d['b'].shape[1]
    ab = np.zeros((d['a'].shape[0], nparam))
    ab[:,0:d['a'].shape[1]] = d['a']
    ab[:,d['a'].shape[1]:] = d['b']

    target = d['Rs']

    dataset = MyDataset(ab)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(nparam, target.shape[1]).to(device)
    train_kwargs = {'batch_size': 32}
