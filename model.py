import torch.nn as nn
import torch.nn.functional as F

class PDnet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # convolutions,batch norms and pools
        self.conv1 = nn.Conv2D(3, 32, 3, padding = 1)
        # 32, 512, 512
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2D(32, 32, 3, padding = 1)
        # 32, 512, 512
        self.bn2 = nn.BatchNorm2d(32)
        # insert pooling layer 32, 256, 256
        self.conv3 = nn.Conv2D(32, 64, 3, padding = 1)
        # 64, 256, 256
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2D(64, 64, 3, padding = 1)
        # 64, 256, 256
        self.bn4 = nn.BatchNorm2d(64)
        # insert pooling layer 64, 128, 128
        self.conv5 = nn.Conv2D(64, 128, 3, padding = 1)
        # 128, 128, 128
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2D(128, 256, 3, padding = 1)
        # 256, 128, 128
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2D(256, 256, 3, padding = 1)
        # 256, 128, 128
        self.bn7 = nn.BatchNorm2d(256)
        # insert pooling layer 256, 64, 64
        self.conv8 = nn.Conv2D(256, 512)
        # 512, 64, 64
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2D(512, 512, 3, padding = 1)
        # 512, 64, 64
        self.bn9 = nn.BatchNorm2d(512)
        # insert pooling layer 512, 32, 32
        self.conv10 = nn.Conv2D(512, 1024, 2, stride = 2, padding = 1)
        self.bn10 = nn.BatchNorm2d(1024)
        # 1024, 16, 16

        # max pooling layer
        self.mpool = nn.MaxPool2d(2,2)
        # average pooling layer
        selg.apool = nn.AvgPool2d(2,2)

        # Fully connected layers
        self.fc1 = nn.Linear(1024*16*16, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 8)

        # dropout layer
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):

        # do the forward pass through the network
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.apool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.apool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.apool(F.relu(self.bn7(self.conv7(x))))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.mpool(F.relu(self.bn9(self.conv9(x))))
        x = F.relu(self.bn10(self.conv10(x)))

        x = x.view(-1, 1024*16*16)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        output = F.relu(self.fc3(x))

        return output
