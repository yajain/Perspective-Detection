import torch.nn as nn
import torch.nn.functional as F

class PDnet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2D(3, 16, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)

        # Add more Conv2d, pooling, batch norm, dropout layers

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # do the forward pass through the network

        return x
