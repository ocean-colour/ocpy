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

# Erdong's Notebook
#   https://github.com/AI-for-Ocean-Science/ulmo/blob/F_S/ulmo/fs_reg_dense/fs_dense_train.ipynb

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
                 nhidden1:int, nhidden2:int,
                 ab_parm:tuple, 
                 Rs_parm:tuple):
        super(SimpleNet, self).__init__()
        # Save
        self.ninput = ninput
        self.noutput = noutput
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2

        self.ab_parm = ab_parm
        self.Rs_parm = Rs_parm

        # Architecture
        self.fc1 = nn.Linear(self.ninput, nhidden1)
        self.fc2 = nn.Linear(nhidden1, noutput)
        self.fc2b = nn.Linear(nhidden1, nhidden2)
        self.fc3 = nn.Linear(nhidden2, noutput)

        # Normalization terms
        

    def forward(self, x):
        '''
        # 1 hidden layers -- 0.23 at 1000
        x = self.fc1(x)
        x = F.relu(x)  # 0.23 at 1000
        #x = F.leaky_relu(x)  # 0.23 at 1000
        #x = F.sigmoid(x)  # Much worse
        x = self.fc2(x)
        '''

        # 2 hidden layers -- 9,17 at 1000
        x = self.fc1(x)
        x = F.relu(x)  
        x = self.fc2b(x)
        x = F.relu(x)  
        x = self.fc3(x)
        #x = F.relu(x)
        #x = self.fc3(x)

        return x

    def prediction(self, sample, device):
        # Normalize the inputs
        norm_sample = (sample - self.ab_parm[0])/self.ab_parm[1]
        tensor = torch.Tensor(norm_sample)

        self.eval()
        with torch.no_grad():
            batch_features = tensor.view(-1, 6).to(device)
            outputs = self(batch_features)

        outputs.cpu()
        pred = outputs * self.Rs_parm[1] + self.Rs_parm[0]

        # Convert to numpy
        pred = pred.numpy()

        return pred.flatten()


def preprocess_data(data):

    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean)/std

    return data.astype(np.float32), mean, std

def perform_training(model, dataset, ishape:int, train_kwargs, lr,
                     nepochs:int=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset, **train_kwargs)

    for epoch in range(nepochs):
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
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, nepochs, loss))

    # Return
    return epoch, loss, optimizer

def build_quick_nn_l23(nepochs:int,
                       root:str='model'):

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
    pre_ab, mean_ab, std_ab = preprocess_data(ab)
    pre_targ, mean_targ, std_targ = preprocess_data(target)

    # Dataset
    dataset = MyDataset(pre_ab, pre_targ)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nhidden1 = 128
    nhidden2 = 128
    # Instantiate
    model = SimpleNet(nparam, target.shape[1], 
                      nhidden1, nhidden2,
                      (mean_ab, std_ab),
                      (mean_targ, std_targ),
                      ).to(device)
    nbatch = 64
    train_kwargs = {'batch_size': nbatch}

    lr = 1e-3
    epoch, loss, optimizer = perform_training(model, dataset, nparam, 
                     train_kwargs, lr, nepochs=nepochs)


    # Save
    PATH = f"{root}.pt"
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
    torch.save(model, f'{root}.pth')
    print(f"Wrote: {root}.pt, {root}.pth")



if __name__ == '__main__':

    # Train
    build_quick_nn_l23(100, root='model_100')
    build_quick_nn_l23(20000, root='model_20000')

    # Test loading and prediction
    test = False
    if test:
        model = torch.load('model_100.pth')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp = model.prediction(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), device)
        embed(header='215 of nn')