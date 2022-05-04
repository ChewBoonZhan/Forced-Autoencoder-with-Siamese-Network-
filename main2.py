import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import random
import math
import cv2
import os
import sys
from sklearn.cluster import KMeans
from tqdm import tqdm
import math
from sklearn.metrics import roc_auc_score, roc_curve
import csv
import argparse

####################################################################
# Import local functions for preprocessing
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Preprocessing/")

from processImage import processImage
from DeviceDataLoader import DeviceDataLoader

####################################################################
# Import local functions for Autoencoder showcase
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Autoencoder_showcase_helper/")
from getImageLabel2 import getImageLabel2  

####################################################################
# Import local functions for Autoencoder training
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Autoencoder_training_helper/Variant2/")
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Autoencoder_training_helper/Variant3/")
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Autoencoder_training_helper/Variant4/")
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Autoencoder_training_helper/Variant5/")
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Autoencoder_training_helper/")

from getEncoderLatentCollection_supervised import getEncoderLatentCollection_supervised
from getImageCollectionLabelTrain import getImageCollectionLabelTrain

from getAutoencoderEncoderRepresentation_supervised import getAutoencoderEncoderRepresentation_supervised

from getEncoderLatentCollection import getEncoderLatentCollection
from getImageForEncoderCollection import getImageForEncoderCollection

from generateLabelRepresentation import generateLabelRepresentation
from getAutoencoderEncoderRepresentation import getAutoencoderEncoderRepresentation
from getImageForEncoderCollection2 import getImageForEncoderCollection2

from check_distance_between_latent import check_distance_between_latent

####################################################################
# Import local functions for KNN Training
sys.path.insert(0, os.getcwd() + "/HelperFunctions/KNN_helper/")

from flattenKNNCollection import flattenKNNCollection

####################################################################
# Import local functions for Siamese accuracy helper
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Siamese_accuracy_helper/")

from checkNovelAccuracy import checkNovelAccuracy
from checkInNovelOrNot import checkInNovelOrNot
from genNovelOrNotOnLabel import genNovelOrNotOnLabel

####################################################################
# Import local functions for Siamese training helper
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Siamese_training_helper/Variant1/")
sys.path.insert(0, os.getcwd() + "/HelperFunctions/Siamese_training_helper/Variant2/")

from getImageLabelTrain import getImageLabelTrain
from genDiffLabelImage_supervised import genDiffLabelImage_supervised
from genImageLabelDataset_supervised import genImageLabelDataset_supervised
from genSameImage_supervised import genSameImage_supervised

from getImageForNoveltyCollection import getImageForNoveltyCollection   
from genSameImage import genSameImage
from genDiffLabelImage import genDiffLabelImage
from genImageLabelDataset import genImageLabelDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################
# Import model architecture
sys.path.insert(0, os.getcwd() + "/Model_architecture/")
from encoder import encoder
from decoder import decoder
from siameseNetwork import siameseNetwork
from ContrastiveLoss import ContrastiveLoss


if __name__ == '__main__':
    #####################################################################
    # Dataset parameters
    # MNIST
    batch_size = 64
    learning_rate = 1e-3
    number_of_channel = 1
    input_image_size = 28
    number_of_layer = 1
    kernel_size = 3

    reduction_size = 0
    if (kernel_size == 3):
        reduction_size = 2
    elif(kernel_size == 5):
        reduction_size = 4

    conv_image_size = input_image_size- (number_of_layer * reduction_size)
    number_of_conv_final_channel = 16
    conv_output_flatten = number_of_conv_final_channel*conv_image_size*conv_image_size

    # latent_space_features = 32   # for reconstruction error
    latent_space_features = 16

    classToUse = [0,1,2,3,4,5,6,7,8,9]
    lengthClassToUse = len(classToUse)

    ## set the novel and familiar class to be random

    novelClassCollection = [4, 5, 6]
    lengthNovelClassCollection = len(novelClassCollection)


    actualNovelClass = [7, 8, 9]
    lengthActualNovelClass = len(actualNovelClass)


    familiarClass = []
    for labelClass in classToUse:
        if((labelClass in novelClassCollection) or (labelClass in actualNovelClass)):
            q = 1
        else:
            familiarClass.append(labelClass)

    totalNumberOfClasses = lengthClassToUse
    totalNumberOfFamiliarity = (lengthClassToUse - lengthNovelClassCollection - lengthActualNovelClass) 
    train_other_loaderClass = familiarClass + novelClassCollection

    # define parser to get argument from command prompt
    parser = argparse.ArgumentParser()
    
    # Get autoencoder variant from user
    parser.add_argument('--autoencoder', nargs='?', default = "1")

    # Get siamese network variant from user
    parser.add_argument('--siamese', nargs='?', default = "1")

    args = parser.parse_args()

    print('Using variance {} autoencoder'.format(args.autoencoder))   
    print('Using variance {} Siamese Network'.format(args.siamese))   

    img = cv2.imread(os.getcwd() + "/Main2_Test_Image/test.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    difference = abs(img.shape[0] - img.shape[1])
    diffDiv = int(difference/2)
    diffDiv2 = difference - diffDiv
    if(img.shape[0] < img.shape[1]):
        # height less than width, pad height
        img = cv2.copyMakeBorder(img, diffDiv, diffDiv2, 0, 0, cv2.BORDER_CONSTANT, None, value = 255)
    else:
        # height more than width
        img = cv2.copyMakeBorder(img, 0, 0, diffDiv, diffDiv2, cv2.BORDER_CONSTANT, None, value = 255)

    # resize to make it same size as dataset image
    img = cv2.resize(img, (input_image_size, input_image_size))

    # threshold it to make it same color as dataset image
    ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # convert it to torch 
    img = torch.tensor(img/255).unsqueeze(0).unsqueeze(0).float()

    # define encoder and decoder
    encoderNet = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)

    decoderNet = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)

    if(args.autoencoder == "6"):
        # autoencoder variant 6
        encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant6Encoder.pth", map_location=device))
        decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant6Decoder.pth", map_location=device))

    elif(args.autoencoder == "2"):
        # autoencoder variant 2
        encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant2Encoder.pth", map_location=device))
        decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant2Decoder.pth", map_location=device))

    elif(args.autoencoder == "3"):
        # autoencoder variant 3
        encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant3Encoder.pth", map_location=device))
        decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant3Decoder.pth", map_location=device))

        

    elif(args.autoencoder == "4"):
        # autoencoder variant 4
        encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant4Encoder.pth", map_location=device))
        decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant4Decoder.pth", map_location=device))

    elif(args.autoencoder == "5"):
        # autoencoder variant 5
        encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant5Encoder.pth", map_location=device))
        decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/variant5Decoder.pth", map_location=device))
        
    else:
        # autoencoder variant 1 (default)
        encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/normalEncoder.pth", map_location=device))
        decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/normalDecoder.pth", map_location=device))

        
    # define siamese network
    novelKNN = siameseNetwork(number_of_conv_final_channel, number_of_channel, conv_output_flatten, kernel_size).float().to(device)

    if(args.siamese == "2"):
        novelKNN.load_state_dict(torch.load(os.getcwd() + "/Data/siameseVariant2.pth", map_location=device))
    else:
        # variant 1 is default
        novelKNN.load_state_dict(torch.load(os.getcwd() + "/Data/siameseVariant1.pth", map_location=device))
        
    # send image into autoencoder
    outImg = decoderNet(encoderNet(img))

    # squeeze output image to image shape
    outImgDisp = outImg.squeeze(0).squeeze(0).detach().cpu().numpy()*255

    # squeeze input image to image shape
    imgDisp = img.squeeze(0).squeeze(0).cpu().detach().numpy()*255

    # declare a new spacing image to add spacing between 2 image
    spacingImage = torch.tensor(np.zeros((imgDisp.shape[0], 10))).float().numpy()
    
    finalImage = cv2.hconcat([imgDisp, spacingImage, outImgDisp])

    # resize to make it same size as dataset image
    finalImage = cv2.resize(finalImage, (finalImage.shape[1]*10, finalImage.shape[0]*10))

    # declare spacing at top of image
    spacingTopImage = torch.tensor(np.zeros((100, finalImage.shape[1]))).float().numpy()

    # concat to top of image
    finalImage = cv2.vconcat([spacingTopImage, finalImage])

    # put text on image for input image and output image
    finalImage = cv2.putText(finalImage, 'Input Image', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    finalImage = cv2.putText(finalImage, 'Output Image', (420, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)

    # declare spacing at bottom of image
    spacingBottomImage = torch.tensor(np.zeros((200, finalImage.shape[1]))).float().numpy()

    # concat to bottom of image
    finalImage = cv2.vconcat([finalImage, spacingBottomImage])

    # estimate novelty of image
    out1, out2 = novelKNN(img,outImg)

    distance = F.pairwise_distance(out1, out2)

    # add similarity between image to image
    finalImage = cv2.putText(finalImage, "Difference between Image: " + "{:.4f}".format(distance[0].item()), (70, 440), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # add novelty to bottom of image
    if(distance[0].item() > 0.5):
        # image is novel
        finalImage = cv2.putText(finalImage, "Novel Image", (240, 490), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        # image is non-novel
        finalImage = cv2.putText(finalImage, "Non-novel Image", (200, 490), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # add label that are non-novel
    finalImage = cv2.putText(finalImage, "Non-novel label: " + str(familiarClass), (120, 540), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)


    plt.imshow(finalImage)
    plt.show()

        

        

    

    
