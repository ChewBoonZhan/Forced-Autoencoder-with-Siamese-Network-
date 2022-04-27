import torch.nn as nn
import torch
class decoder(nn.Module):
    def __init__(self, latent_space_features, number_of_conv_final_channel, conv_image_size, imgChannels, featureDim, kernel_size):
        super(decoder, self).__init__()

        self.number_of_conv_final_channel = number_of_conv_final_channel
        self.conv_image_size = conv_image_size
    
        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.d1 = nn.Linear(latent_space_features, featureDim)

        self.decConv1 = nn.ConvTranspose2d(number_of_conv_final_channel, imgChannels, kernel_size)


    def forward(self, x):

        x = torch.tanh(self.d1(x))       
       
        x = x.view(-1, self.number_of_conv_final_channel, self.conv_image_size, self.conv_image_size)
        x = torch.sigmoid(self.decConv1(x))
        return x