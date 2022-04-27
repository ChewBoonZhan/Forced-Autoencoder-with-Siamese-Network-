import torch
import torch.nn as nn
class siameseNetwork(nn.Module):
    def __init__(self, number_of_conv_final_channel, imgChannels, featureDim, kernel_size):
        super(siameseNetwork, self).__init__()
        self.featureDim = featureDim
        self.encConv1 = nn.Conv2d(imgChannels, number_of_conv_final_channel, kernel_size)   # kernel size is 5

        self.l1 = nn.Linear(featureDim, 2048)
        
        
        

    def forward(self, x,x2):

        x = torch.relu(self.encConv1(x))
        x = x.view(-1, self.featureDim)
        x = torch.sigmoid((self.l1(x)))

        x2 = torch.relu(self.encConv1(x2))
        x2 = x2.view(-1, self.featureDim)
        x2 = torch.sigmoid((self.l1(x2)))

        return (x, x2)
