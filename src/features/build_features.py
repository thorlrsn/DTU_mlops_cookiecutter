from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class MyAwesomeModel(nn.Module):
    # def __init__(self):
    #     super(MyAwesomeModel, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    #     self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
    #     self.conv3 = nn.Conv2d(32,64, kernel_size=5)
    #     self.fc1 = nn.Linear(3*3*64, 256)
    #     self.fc2 = nn.Linear(256, 10)
        
    # def forward(self, x):
    #     # catching dimension errors.
    #     if x.ndim != 4:
    #         raise ValueError('Expected input to a 4D tensor')
    #     if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
    #         raise ValueError('Expected each sample to have shape [1, 28, 28]')
        
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(F.max_pool2d(self.conv2(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv3(x),2))
    #     x = x.view(-1,3*3*64 )
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return F.log_softmax(x, dim=1)
    def __init__(self):
        super().__init__()
        # torchvision.models.ResNet18_Weights
        self.network = models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 120),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, xb):
        return self.network(xb)