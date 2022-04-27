import torch.nn as nn
import torch
class encoder(nn.Module):
    def __init__(self, number_of_conv_final_channel, latent_space_features, imgChannels, featureDim, kernel_size):
        super(encoder, self).__init__()

        self.featureDim = featureDim

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, number_of_conv_final_channel, kernel_size)   # kernel size is 5
        
        self.l1 = nn.Linear(featureDim, latent_space_features)
        

    def forward(self, x):

        x = torch.relu(self.encConv1(x))
        
        x = x.view(-1, self.featureDim)
        
        x = torch.tanh(self.l1(x))
        return x