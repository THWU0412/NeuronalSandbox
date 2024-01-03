"""
    DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ
    Link: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
    3-layered neuronal network with 32*32 input and 10 output nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

import sys

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3-layered Neural Network
        self.fc1 = nn.Linear(32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create necessary resources
net = Net()
input = torch.randn(1, 1, 32, 32)
target = torch.randn(1, 10)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

def update(frame):
    plt.cla()                   # Clear the diagram

    optimizer.zero_grad()       # Zeroes the gradient buffers of all parameters

    out = net(input)            # Use the neuronal network

    loss = criterion(out, target)   # Calculate the loss

    if loss.data < 0.0001:          # Check if the loss is below a certain threshold and end the training
        print('Finished in ' + str(frame+1) + ' Iterations')
        sys.exit()

    loss.backward()             # Calculate the new gradients

    optimizer.step()            # Update weights
    
    plt.plot(out.detach().numpy()[0])   # Plot the current function
    plt.plot(target.numpy()[0])         # Plot the target function


# Create the empty diagram
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, cache_frame_data=False, interval=100)

plt.show()