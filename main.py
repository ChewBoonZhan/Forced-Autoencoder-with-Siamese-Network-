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
#####################################################################

## Loading dataset
from keras.datasets import mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#####################################################################
# Processing dataset for later use

# Testing dataset, contains all labels except for actual novel
print("\nProcessing Testing dataset 1, contains novel and non-novel labels")
test_loader_no_actual, test_label_no_actual = processImage(test_X, 
                                       test_Y, 
                                       removeNovel = False,
                                       removeActualNovel = True, 
                                       novelClassCollection = novelClassCollection,
                                       actualNovelClass = actualNovelClass,
                                       classToUse = classToUse
                                       )

# Testing dataset, contains all labels
print("\nProcessing Testing dataset 2, contains all labels, ie actual, novel and non-novel labels")
test_loader, test_label = processImage(test_X, 
                                       test_Y, 
                                       removeNovel = False,
                                       removeActualNovel = False, 
                                       novelClassCollection = novelClassCollection,
                                       actualNovelClass = actualNovelClass,
                                       classToUse = classToUse
                                       )



# Training dataset for autoencoder, contains non-novel label only
print("\nProcessing Train dataset 1, contains non-novel labels")
trainData, trainLabel = processImage(train_X, 
                                     train_Y, 
                                     removeNovel = True,
                                     removeActualNovel = True,
                                     novelClassCollection = novelClassCollection,
                                     actualNovelClass = actualNovelClass,
                                     classToUse = classToUse
                                    )

# Training dataset for Siamese Network, contains novel label and non-novel label. Does not have actual novel label
print("\nProcessing Train dataset 2, contains novel and non-novel labels")
trainOtherData, trainOtherLabel = processImage(train_X, 
                                               train_Y, 
                                               removeNovel = False,
                                               removeActualNovel = True,
                                               novelClassCollection = novelClassCollection,
                                               actualNovelClass = actualNovelClass,
                                               classToUse = classToUse
                                            )

#####################################################################
# Convert data into tensor
# Test Data for method 2
test_loader = torch.tensor(test_loader)
test_label = torch.tensor(test_label)

# Train Data for Autoencoder
trainData = torch.tensor(trainData)
trainLabel = torch.tensor(trainLabel)

# Train Data for Siamese Network
trainOtherData = torch.tensor(trainOtherData)
trainOtherLabel = torch.tensor(trainOtherLabel)

# Test data for method 1
test_loader_no_actual = torch.tensor(test_loader_no_actual)
test_label_no_actual = torch.tensor(test_label_no_actual)

#####################################################################
# Convert tensor into tensor dataset

# Train data for Autoencoder
trainData2 = TensorDataset(trainData, trainLabel)

# Train data for Siamese Network
trainOtherData = TensorDataset(trainOtherData, trainOtherLabel)

# Test Data for Method 2
test_loader = TensorDataset(test_loader, test_label)

# Test Data for Method 2
test_loader_no_actual= TensorDataset(test_loader_no_actual, test_label_no_actual)

#####################################################################
# Convert tensor dataset into Dataloader, make them into batch sizes, and shuffle

# Train data for Autoencoder
train_loader = DataLoader(trainData2, batch_size, shuffle=True)

# Train data for Siamese Network
train_other_loader = DataLoader(trainOtherData, batch_size, shuffle=True)

#####################################################################
# make dataloader to device

# Train data for Autoencoder
train_loader = DeviceDataLoader(train_loader, device)

# Train data for Siamese Network
train_other_loader = DeviceDataLoader(train_other_loader, device)

# Test data with actual novel label, novel label and non-novel label - method 1
test_loader = DeviceDataLoader(test_loader, device)

# Test data with novel label and non-novel label - method 2
test_loader_no_actual= DeviceDataLoader(test_loader_no_actual, device)

#####################################################################
# get knnImage and knnLabel
print("\nPreparing dataset for KNN for Variant 4 Autoencoder\n")
knnImage = []
oriImage = []

for data in trainData2:
  
  img, label  = data

  imageTemp = img.to("cpu").numpy()
  
  flattenedShape = imageTemp.shape[1] * imageTemp.shape[2]
  imageTemp = np.reshape(imageTemp, (flattenedShape))

  knnImage.append(imageTemp.tolist())
  oriImage.append(img)

#####################################################################
# Define and Train the first KNN Model
numberOfClusters = 4

print("\nTraining KNN For Variant 4 Autoencoder\n")

## as of now the number of clusters depend highly on the number of labels used

knnModel = KMeans(n_clusters=numberOfClusters)

knnModel.fit(knnImage)

#####################################################################
# Define function to get image from KNN 1 to train Autoencoder
counter = 0
autoencoderTrainImage = []
for index in range(numberOfClusters):
  while True:
    knnImageIndex = knnImage[counter]
    oriImageIndex = oriImage[counter]

    predictedValue = knnModel.predict([knnImageIndex])
    if(predictedValue == index):
      autoencoderTrainImage.append(oriImageIndex.to("cpu").numpy())
      break

    counter = counter + 1

#####################################################################
# Define and Train the second KNN Model
# get knnImage and knnLabel
knnImage2 = []
oriImage2 = []

for data in trainOtherData:
  img, label  = data

  imageTemp = img.to("cpu").numpy()
  
  flattenedShape = imageTemp.shape[1] * imageTemp.shape[2]
  imageTemp = np.reshape(imageTemp, (flattenedShape))

  knnImage2.append(imageTemp.tolist())
  oriImage2.append(img)

numberOfClusters2 = 7

## as of now the number of clusters depend highly on the number of labels used

print("\nTraining KNN For Siamese Network Variant 2\n")
knnModel2 = KMeans(n_clusters=numberOfClusters2)

knnModel2.fit(knnImage2)

#####################################################################
# Define function to get image from KNN 2 
print("\nPreparing dataset to get image from KNN for Siamese Network Variant 2\n")
noveltyTrainImage = []
for index in range(numberOfClusters2):
  counter = 0
  numValue = 50
  noveltyTrainImageIndex = []
  
  while True:
    knnImageIndex = knnImage2[counter]
    oriImageIndex = oriImage2[counter]
    counter = counter + 1
    predictedValue = knnModel2.predict([knnImageIndex])
    if(predictedValue[0] == index):
      noveltyTrainImageIndex.append(oriImageIndex.to("cpu").numpy())
      numValue = numValue - 1
      if(numValue == 0):
        noveltyTrainImage.append(noveltyTrainImageIndex)
        break

#####################################################################
# Define function to get a single image from test image dataset
imageCollectionForAllIndex = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
}
imageCollectionCountForAllIndex = [
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100
]
for image, label in trainOtherData:
  labelCount = imageCollectionCountForAllIndex[label.item()]
  if(labelCount == 0):
    continue
  else:
    # there is still count left
    imageCollectionCountForAllIndex[label.item()] = imageCollectionCountForAllIndex[label.item()] - 1

    arrayIndex = imageCollectionForAllIndex[label.item()]

    arrayIndex.append(image)

    imageCollectionForAllIndex[label.item()] = arrayIndex

imageCollectionForIndex = []
otherClassLength = len(train_other_loaderClass)
for index in range(otherClassLength):
  index = train_other_loaderClass[index]
  tempImage = getImageLabelTrain(index,1, imageCollectionForAllIndex)
  # invert the image
  # tempImageOnes = torch.tensor(np.ones(tempImage.shape))
  # tempImage = tempImageOnes - tempImage
  imageCollectionForIndex.append(tempImage.to("cpu").numpy())

#####################################################################
# Define the forced encoder representation
print("\nDefining Forced Encoder Representation for Variant 2 and Variant 4 Autoencoder\n")
encoderLatentCollection = []
index = 0

while index  < (numberOfClusters):
  tempRandomNumberCollection = (np.random.normal(0,1,latent_space_features).tolist())   
  if(check_distance_between_latent(tempRandomNumberCollection, encoderLatentCollection, latent_space_features)):
    print("1 Completed")
    encoderLatentCollection.append(tempRandomNumberCollection)
    index = index + 1
  else:
    continue


#####################################################################
# Define Encoder and Decoder
## Autoencoder for KNN training for Siamese Network
print("\nDeclaring Autoencoders\n")
encoderNet0 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp0 = torch.optim.Adam(encoderNet0.parameters(), lr=learning_rate)

decoderNet0 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp0 = torch.optim.Adam(decoderNet0.parameters(), lr=learning_rate)

## Autoencoder for evaluating performance, trained in normal fashion 
encoderNet = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp = torch.optim.Adam(encoderNet.parameters(), lr=learning_rate)

decoderNet = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp = torch.optim.Adam(decoderNet.parameters(), lr=learning_rate)

## KNN Force Autoencoder Collection  
encoderNet2 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp2 = torch.optim.Adam(encoderNet2.parameters(), lr=learning_rate)

decoderNet2 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp2 = torch.optim.Adam(decoderNet2.parameters(), lr=learning_rate)

encoderNet3 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp3 = torch.optim.Adam(encoderNet3.parameters(), lr=learning_rate)

decoderNet3 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp3 = torch.optim.Adam(decoderNet3.parameters(), lr=learning_rate)

encoderNet4 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp4 = torch.optim.Adam(encoderNet4.parameters(), lr=learning_rate)

decoderNet4 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp4 = torch.optim.Adam(decoderNet4.parameters(), lr=learning_rate)

# Supervised Force Autoencoder COllection
encoderNet_supervised = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp_supervised = torch.optim.Adam(encoderNet_supervised.parameters(), lr=learning_rate)

decoderNet_supervised = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp_supervised = torch.optim.Adam(decoderNet_supervised.parameters(), lr=learning_rate)

encoderNet2_supervised = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp2_supervised = torch.optim.Adam(encoderNet2_supervised.parameters(), lr=learning_rate)

decoderNet2_supervised = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp2_supervised = torch.optim.Adam(decoderNet2_supervised.parameters(), lr=learning_rate)


#####################################################################
# Loading/Training Autoencoder for KNN 
if(os.path.exists(os.getcwd() + "/Data/encoderTrainedKNN.pth") and os.path.exists(os.getcwd() + "/Data/decoderTrainedKNN.pth")):
  print("\nLoading encoder and decoder for KNN for variant 5 Autoencoder\n")
  encoderNet0.load_state_dict(torch.load(os.getcwd() + "/Data/encoderTrainedKNN.pth", map_location=device))
  decoderNet0.load_state_dict(torch.load(os.getcwd() + "/Data/decoderTrainedKNN.pth", map_location=device))
else:
  print("\nTraining encoder and decoder for KNN for variant 5 Autoencoder\n")
  num_epochs = 40
  retrainDecoderTimes = 100


  for epoch in tqdm(range(0, num_epochs), desc ="Training Autoencoder for KNN"):

    for idx, data in enumerate(train_loader, 0):
        # new line
        # set the model into a train state
        encoderNet0.train()
        decoderNet0.train()
      
        imgs, _ = data

        imgs = imgs.float()

      
        outEncoder = encoderNet0(imgs)
        outDecoder = decoderNet0(outEncoder)
        lossAutoencoder = nn.functional.mse_loss(outDecoder, imgs)
        
        encoderOp0.zero_grad()
        decoderOp0.zero_grad()
        lossAutoencoder.backward()
        encoderOp0.step()
        decoderOp0.step()

    print('\nEpoch {}: Autoencoder Loss {}'.format(epoch, lossAutoencoder))   

  # saving model
  torch.save(encoderOp0.state_dict(), os.getcwd() + "/Data/encoderTrainedKNN.pth")
  torch.save(decoderOp0.state_dict(), os.getcwd() + "/Data/decoderTrainedKNN.pth")

#####################################################################
# Train KNN 3 to get representation from encoder
outputEncoderCollectionKNN = []

for data in trainData2:
  
  img, label  = data

  img = img.unsqueeze(0).to(device).float()

  outputEncoder = encoderNet0(img)
  
  outputEncoderCollectionKNN.append(outputEncoder.detach().to("cpu").squeeze(0).numpy().tolist())

numberOfClusters3 = 4

print("\nTraining KNN for variant 5 Autoencoder using latent space from normal autoencoder\n")
knnModel3 = KMeans(n_clusters=numberOfClusters3)

knnModel3.fit(outputEncoderCollectionKNN)


#####################################################################
# Preparing dataset for training of autoencoder variants..
print("\nPreparing dataset for training of autoencoder variants..\n")
autoencoderEncoderCollection = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
}
for data in trainData2:
  
  img, label  = data

  img = img.unsqueeze(0).to(device).float()

  outputEncoder = encoderNet0(img.float())

  outputEncoder = outputEncoder.detach().to("cpu").squeeze(0).numpy().tolist()

  predictedValue = knnModel3.predict([outputEncoder])

  autoencoderEncoderCollection[predictedValue[0]].append(outputEncoder)

#####################################################################
# Generating encoder collection for Variant 5 Autoencoder

autoencoderEncoderCollectionAvg = []
for index in range(10): 
  encoderAutoencoderCollectionIndex = torch.tensor(autoencoderEncoderCollection[index])
  averageEncoder = torch.mean(encoderAutoencoderCollectionIndex, axis = 0)
  # print(averageEncoder.shape)
  if(averageEncoder.shape == torch.tensor(float("nan")).shape):
    q=1
  else:
    autoencoderEncoderCollectionAvg.append(averageEncoder.squeeze(0).numpy().tolist())


#####################################################################
# Generating train image for Variant 5 Autoencoder
counter = 0
autoencoderTrainImage2 = []
for index in range(numberOfClusters3):
  while True:
    oriImageIndex = oriImage[counter]

    oriImageIndex = oriImageIndex.unsqueeze(0).to(device).float()
    outputEncoder = encoderNet0(oriImageIndex)

    outputEncoder = outputEncoder.detach().to("cpu").float().numpy().astype(np.float64)


    predictedLabel = knnModel3.predict(outputEncoder)

    if(predictedLabel[0] == index):
      autoencoderTrainImage2.append(oriImageIndex.to("cpu").numpy())
      break


    counter = counter + 1

#####################################################################
# Get average latent space of autoencoder for training Variant 3 autoencoder
encoderAutoencoderCollection = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[],
}
encoderAutoencoderRepresentation = []
for data in trainData2:
  encoderNet0.eval()
  decoderNet0.eval()
  
  img, label = data
  img = img.unsqueeze(0).float().to(device)

  outEncoder = encoderNet0(img)
      
  encoderAutoencoderCollection[label.item()].append(outEncoder.detach().to("cpu").numpy().tolist())

for index in range(10):
  encoderAutoencoderCollectionIndex = torch.tensor(encoderAutoencoderCollection[index])
  averageEncoder = torch.mean(encoderAutoencoderCollectionIndex, axis = 0)
  # print(averageEncoder.shape)
  if(averageEncoder.shape == torch.tensor(float("nan")).shape):
    q=1
  else:
    encoderAutoencoderRepresentation.append(averageEncoder.squeeze(0).numpy().tolist())

#####################################################################
# Training autoencoder
if(os.path.exists(os.getcwd() + "/Data/normalEncoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/normalDecoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/variant4Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/variant4Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/variant5Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/variant5Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/variant6Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/variant6Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/variant2Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/variant2Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/variant3Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/variant3Decoder.pth")
    ):
  print("\nLoading Autoencoder\n")
  encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/normalEncoder.pth", map_location=device))
  decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/normalDecoder.pth", map_location=device))

  encoderNet2.load_state_dict(torch.load(os.getcwd() + "/Data/variant4Encoder.pth", map_location=device))
  decoderNet2.load_state_dict(torch.load(os.getcwd() + "/Data/variant4Decoder.pth", map_location=device))

  encoderNet3.load_state_dict(torch.load(os.getcwd() + "/Data/variant5Encoder.pth", map_location=device))
  decoderNet3.load_state_dict(torch.load(os.getcwd() + "/Data/variant5Decoder.pth", map_location=device))

  encoderNet4.load_state_dict(torch.load(os.getcwd() + "/Data/variant6Encoder.pth", map_location=device))
  decoderNet4.load_state_dict(torch.load(os.getcwd() + "/Data/variant6Decoder.pth", map_location=device))

  encoderNet_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/variant2Encoder.pth", map_location=device))
  decoderNet_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/variant2Decoder.pth", map_location=device))

  encoderNet2_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/variant3Encoder.pth", map_location=device))
  decoderNet2_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/variant3Decoder.pth", map_location=device))

else:
  print("\nTraining Autoencoder\n")
  num_epochs = 40
  retrainDecoderTimes = 100


  for epoch in tqdm(range(0, num_epochs), desc ="Training Autoencoder"):

    for idx, data in enumerate(train_loader, 0):
        # new line
        # set the model into a train state
        encoderNet.train()
        decoderNet.train()

        encoderNet2.train()
        decoderNet2.train()

        encoderNet3.train()
        decoderNet3.train()

        encoderNet4.train()
        decoderNet4.train()

        encoderNet_supervised.train()
        decoderNet_supervised.train()

        encoderNet2_supervised.train()
        decoderNet2_supervised.train()


        imgs, actualLabel = data


        # KNN Variant 4 - perform KNN in the image to get label for images
        flattenedImage = flattenKNNCollection(imgs)

        label = knnModel.predict(flattenedImage)

        # get the latent representation for the associated label   
        encoderRepresentation = getEncoderLatentCollection(label, encoderLatentCollection, device)
        
        imgs = imgs.float()

        imagesCompare = getImageForEncoderCollection(label, autoencoderTrainImage, device).float()
      
        outEncoder = encoderNet2(imgs)

        lossEncoder_variant4 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp2.zero_grad()
        lossEncoder_variant4.backward()
        encoderOp2.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet2(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet2(outEncoder)
        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant4 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp2.zero_grad()
        lossDecoder_variant4.backward()
        decoderOp2.step()
        
        # Train the autoencoder Variant 1
        outEncoder = encoderNet(imgs)
        outDecoder = decoderNet(outEncoder)

        lossAutoencoder = nn.functional.mse_loss(outDecoder, imgs)

        encoderOp.zero_grad()
        decoderOp.zero_grad()
        lossAutoencoder.backward()
        encoderOp.step()
        decoderOp.step()


        # Train the KNN Encoder Autoencoder Variant 5 - use latent space to KNN to produce label train autoencoder
        labelsCollection = generateLabelRepresentation(imgs, encoderNet0, knnModel3)
        
        encoderRepresentation = getAutoencoderEncoderRepresentation(labelsCollection, autoencoderEncoderCollectionAvg, device)

        imageCollection = getImageForEncoderCollection2(labelsCollection, autoencoderTrainImage2, device)

        outEncoder = encoderNet3(imgs)

        lossEncoder_variant5 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp3.zero_grad()
        lossEncoder_variant5.backward()
        encoderOp3.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet3(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet3(outEncoder)
        imagesCompare = ((0.9 * imageCollection) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant5 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp3.zero_grad()
        lossDecoder_variant5.backward()
        decoderOp3.step()

        # Train the new autoencoder variant 6
        labelsCollection = generateLabelRepresentation(imgs, encoderNet0, knnModel3)
        
        encoderRepresentation = getEncoderLatentCollection(labelsCollection, encoderLatentCollection, device)  # change here

        imageCollection = getImageForEncoderCollection2(labelsCollection, autoencoderTrainImage2, device)

        outEncoder = encoderNet4(imgs)

        lossEncoder_variant6 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp4.zero_grad()
        lossEncoder_variant6.backward()
        encoderOp4.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet4(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet4(outEncoder)
        imagesCompare = ((0.9 * imageCollection) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant6 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp4.zero_grad()
        lossDecoder_variant6.backward()
        decoderOp4.step()


        ############ train supervised autoencoder
        # variant 2
        # get the random latent representation for the associated label
        encoderRepresentation = getEncoderLatentCollection_supervised(actualLabel, familiarClass, encoderLatentCollection, device)
        imagesCompare = getImageCollectionLabelTrain(actualLabel, train_other_loaderClass, imageCollectionForIndex, device).float()

        outEncoder = encoderNet_supervised(imgs)

        lossEncoder_variant2 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp_supervised.zero_grad()
        lossEncoder_variant2.backward()
        encoderOp_supervised.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet_supervised(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet_supervised(outEncoder)
        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant2 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp_supervised.zero_grad()
        lossDecoder_variant2.backward()
        decoderOp_supervised.step()


        # Train the autoencoder variant 3 - use latent space from autoencoder averaged
        encoderRepresentation = getAutoencoderEncoderRepresentation_supervised(actualLabel, familiarClass, encoderAutoencoderRepresentation, device)

        outEncoder = encoderNet2_supervised(imgs)
        lossEncoder_variant3 = nn.functional.mse_loss(outEncoder, encoderRepresentation)

        encoderOp2_supervised.zero_grad()
        lossEncoder_variant3.backward()
        encoderOp2_supervised.step()


        outEncoder = encoderNet2_supervised(imgs)
        outEncoder = outEncoder.detach()
        outDecoder = decoderNet2_supervised(outEncoder)

        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant3 = nn.functional.mse_loss(outDecoder, imagesCompare)

        decoderOp2_supervised.zero_grad()
        lossDecoder_variant3.backward()
        decoderOp2_supervised.step()



    
    print('\nEpoch {}: Variant 1 Loss {}'.format(epoch, lossAutoencoder))   
    print('Epoch {}: Variant 2 Encoder Loss {}'.format(epoch, lossEncoder_variant2))
    print('Epoch {}: Variant 2 Decoder Loss {}'.format(epoch, lossDecoder_variant2))

    print('Epoch {}: Variant 3 Encoder Loss {}'.format(epoch, lossEncoder_variant3))
    print('Epoch {}: Variant 3 Decoder Loss {}'.format(epoch, lossDecoder_variant3))

    print('Epoch {}: Variant 4 Encoder Loss {}'.format(epoch, lossEncoder_variant4))
    print('Epoch {}: Variant 4 Decoder Loss {}'.format(epoch, lossDecoder_variant4))

    print('Epoch {}: Variant 5 Encoder Loss {}'.format(epoch, lossEncoder_variant5))
    print('Epoch {}: Variant 5 Decoder Loss {}'.format(epoch, lossDecoder_variant5))

    print('Epoch {}: Variant 6 Encoder Loss {}'.format(epoch, lossEncoder_variant6))
    print('Epoch {}: Variant 6 Decoder Loss {}'.format(epoch, lossDecoder_variant6))


  
  # saving model
  torch.save(encoderNet.state_dict(), os.getcwd() + "/Data/normalEncoder.pth")
  torch.save(decoderNet.state_dict(), os.getcwd() + "/Data/normalDecoder.pth")

  torch.save(encoderNet2.state_dict(), os.getcwd() + "/Data/variant4Encoder.pth")
  torch.save(decoderNet2.state_dict(), os.getcwd() + "/Data/variant4Decoder.pth")

  torch.save(encoderNet3.state_dict(), os.getcwd() + "/Data/variant5Encoder.pth")
  torch.save(decoderNet3.state_dict(), os.getcwd() + "/Data/variant5Decoder.pth")

  torch.save(encoderNet4.state_dict(), os.getcwd() + "/Data/variant6Encoder.pth")
  torch.save(decoderNet4.state_dict(), os.getcwd() + "/Data/variant6Decoder.pth")

  torch.save(encoderNet_supervised.state_dict(), os.getcwd() + "/Data/variant2Encoder.pth")
  torch.save(decoderNet_supervised.state_dict(), os.getcwd() + "/Data/variant2Decoder.pth")

  torch.save(encoderNet2_supervised.state_dict(), os.getcwd() + "/Data/variant3Encoder.pth")
  torch.save(decoderNet2_supervised.state_dict(), os.getcwd() + "/Data/variant3Decoder.pth")

#####################################################################
# Save Variant 1Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 1 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "1"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet.eval()
      decoderNet.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet(outputModelIn)
      outputModel = decoderNet(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 2 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 2 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "2"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet_supervised.eval()
      decoderNet_supervised.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet_supervised(outputModelIn)
      outputModel = decoderNet_supervised(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 3 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 3 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "3"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet2_supervised .eval()
      decoderNet2_supervised .eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet2_supervised (outputModelIn)
      outputModel = decoderNet2_supervised (outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 4 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 4 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "4"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet2.eval()
      decoderNet2.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet2(outputModelIn)
      outputModel = decoderNet2(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 5 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 5 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "5"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet3.eval()
      decoderNet3.eval()
          
      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet3(outputModelIn)
      outputModel = decoderNet3(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 6 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 6 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "6"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet4.eval()
      decoderNet4.eval()
          
      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet4(outputModelIn)
      outputModel = decoderNet4(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train1/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)


#####################################################################
# Define Siamese Network and Contrastive Loss for training
novelKNN = siameseNetwork(number_of_conv_final_channel, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
novelOptimizerKNN = torch.optim.Adam(novelKNN.parameters(), lr=learning_rate)

novelSupervised= siameseNetwork(number_of_conv_final_channel, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
novelOptimizerSupervised = torch.optim.Adam(novelSupervised.parameters(), lr=learning_rate)

lossFunction1 = ContrastiveLoss()


# #####################################################################
# # Training/Loading Siamese Network using Contrastive Loss
if(os.path.exists(os.getcwd() + "/Data/siameseVariant2.pth") and 
    os.path.exists(os.getcwd() + "/Data/siameseVariant1.pth")):
  print("\nLoading Siamese Network\n")
  novelKNN.load_state_dict(torch.load(os.getcwd() + "/Data/siameseVariant2.pth", map_location=device))
  novelSupervised.load_state_dict(torch.load(os.getcwd() + "/Data/siameseVariant1.pth", map_location=device))
else:
  print("\nTraining Siamese Network\n")
    
  num_epochs = 30

  novel_loss = []
  novel_loss2 = []


  for epoch in tqdm(range(0, num_epochs), desc ="Training Siamese Network"):
    
    for idx, data in enumerate(train_other_loader, 0):

        novelKNN.train()

        novelSupervised.train()

        imgs, actualLabel = data
        imgs = imgs.float()

        ## Train KNN siamese network

        imgsTrain1, imgsTrain2, labelTrain = genImageLabelDataset(imgs, knnModel2, noveltyTrainImage, numberOfClusters2, device)

        labelTrain = labelTrain.unsqueeze(0).permute(1,0).float().to(device)
        
        out1, out2 = novelKNN(imgsTrain1,imgsTrain2)

        lossKNN = lossFunction1(out1, out2, labelTrain)

        novelOptimizerKNN.zero_grad()
        lossKNN.backward()
        novelOptimizerKNN.step()

        ## Train supervised siamese network

        imgsTrain1, imgsTrain2, labelTrain = genImageLabelDataset_supervised(imgs, actualLabel, device, imageCollectionForAllIndex, train_other_loaderClass)

        labelTrain = labelTrain.unsqueeze(0).permute(1,0).float().to(device)
        
        out1, out2 = novelSupervised(imgsTrain1,imgsTrain2)

        lossSupervised = lossFunction1(out1, out2, labelTrain)

        novelOptimizerSupervised.zero_grad()
        lossSupervised.backward()
        novelOptimizerSupervised.step()


    print('Epoch {}: Loss for Siamese KNN {}'.format(epoch, lossKNN))
    print('Epoch {}: Loss for SIamese sup {}'.format(epoch, lossSupervised))

  # save models
  torch.save(novelKNN.state_dict(), os.getcwd() + "/Data/siameseVariant2.pth")
  torch.save(novelSupervised.state_dict(), os.getcwd() + "/Data/siameseVariant1.pth")

#####################################################################
# Check accuracy of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# A1S1
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A1S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A1S1 Accuracy: %f"%A1S1Accuracy)


# A1S2
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A1S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A1S2 Accuracy: %f"%A1S2Accuracy)

# A2S1
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)  

    imageIn = decoderNet_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A2S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A2S1 Accuracy: %f"%A2S1Accuracy)

# A2S2
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)

    imageIn = decoderNet_supervised(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A2S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A2S2 Accuracy: %f"%A2S2Accuracy)

# A3S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A3S1 Accuracy: %f"%A3S1Accuracy)

# A3S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A3S2 Accuracy: %f"%A3S2Accuracy)

# A4S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A4S1 Accuracy: %f"%A4S1Accuracy)


# A4S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A4S2 Accuracy: %f"%A4S2Accuracy)


# A5S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A5S1 Accuracy: %f"%A5S1Accuracy)


# A5S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A5S2 Accuracy: %f"%A5S2Accuracy)

# A6S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A6S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A6S1 Accuracy: %f"%A6S1Accuracy)


# A6S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A6S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A6S2 Accuracy: %f"%A6S2Accuracy)

#####################################################################
# Check AUROC of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# A1S1

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S1FPR, A1S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)
print("Method 1 A1S1 AUROC: %f"%A1S1AUROC)

# A1S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S2FPR, A1S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A1S2 AUROC: %f"%A1S2AUROC)

# A2S1

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S1FPR, A2S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A2S1 AUROC: %f"%A2S1AUROC)

# A2S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised.eval()
    decoderNet_supervised.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S2FPR, A2S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A2S2 AUROC: %f"%A2S2AUROC)

# A3S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S1FPR, A3S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A3S1 AUROC: %f"%A3S1AUROC)

# A3S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S2FPR, A3S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A3S2 AUROC: %f"%A3S2AUROC)

# A4S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S1FPR, A4S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A4S1 AUROC: %f"%A4S1AUROC)

# A4S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S2FPR, A4S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A4S2 AUROC: %f"%A4S2AUROC)

# A5S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S1AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S1FPR, A5S1TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A5S1 AUROC: %f"%A5S1AUROC)

# A5S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S2AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S2FPR, A5S2TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A5S2 AUROC: %f"%A5S2AUROC)

# A6S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S1AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S1FPR, A6S1TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A6S1 AUROC: %f"%A6S1AUROC)

# A6S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S2AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S2FPR, A6S2TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A6S2 AUROC: %f"%A6S2AUROC)

#####################################################################
# Generate AUROC graph
print("\nGenerating AUROC graph for Method 1\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Method_1_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Method_1_AUROC/")

plt.title("ROC Curve Actual Novel Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(A1S1FPR, A1S1TPR, label = "A1S1")
plt.plot(A1S2FPR, A1S2TPR, label = "A1S2")
plt.plot(A2S1FPR, A2S1TPR, label = "A2S1")
plt.plot(A2S2FPR, A2S2TPR, label = "A2S2")
plt.plot(A3S1FPR, A3S1TPR, label = "A3S1")
plt.plot(A3S2FPR, A3S2TPR, label = "A3S2")
plt.plot(A4S1FPR, A4S1TPR, label = "A4S1")
plt.plot(A4S2FPR, A4S2TPR, label = "A4S2")
plt.plot(A5S1FPR, A5S1TPR, label = "A5S1")
plt.plot(A5S2FPR, A5S2TPR, label = "A5S2")
plt.plot(A6S1FPR, A6S1TPR, label = "A6S1")
plt.plot(A6S2FPR, A6S2TPR, label = "A6S2")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train1/Method_1_AUROC/method_1_AUROC.png")
plt.clf()

#####################################################################
# Check accuracy of variants (Method 2)
# Method 2 consist of novel class and non-novel class only. It does not contain actual novel class
# The label used for testing is similar with the label used to train Siamese Network
# The data are different, with testing and training dataset

# A1S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A1S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A1S1 Accuracy: %f"%A1S1NovelAccuracy)

# A1S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A1S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A1S2 Accuracy: %f"%A1S2NovelAccuracy)

# A2S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)  

    imageIn = decoderNet_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A2S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A2S1 Accuracy: %f"%A2S1NovelAccuracy)

# A2S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)

    imageIn = decoderNet_supervised(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A2S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A2S2 Accuracy: %f"%A2S2NovelAccuracy)

# A3S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A3S1 Accuracy: %f"%A3S1NovelAccuracy)

# A3S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A3S2 Accuracy: %f"%A3S2NovelAccuracy)

# A4S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A4S1 Accuracy: %f"%A4S1NovelAccuracy)

# A4S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A4S2 Accuracy: %f"%A4S2NovelAccuracy)

# A5S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A5S1 Accuracy: %f"%A5S1NovelAccuracy)

# A5S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A5S2 Accuracy: %f"%A5S2NovelAccuracy)

# A6S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A6S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A6S1 Accuracy: %f"%A6S1NovelAccuracy)


# A6S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A6S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A6S2 Accuracy: %f"%A6S2NovelAccuracy)

#####################################################################
# Get AUROC for variants


# A1S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S1NovelFPR, A1S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A1S1 AUROC: %f"%A1S1NovelAUROC)


# A1S2

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S2NovelFPR, A1S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A1S2 AUROC: %f"%A1S2NovelAUROC)

# A2S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S1NovelFPR, A2S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 3 AUROC: %f"%A2S1NovelAUROC)


# A2S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised.eval()
    decoderNet_supervised.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S2NovelFPR, A2S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A2S2 AUROC: %f"%A2S2NovelAUROC)

# A3S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S1NovelFPR, A3S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A3S1 AUROC: %f"%A3S1NovelAUROC)


# A3S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S2NovelFPR, A3S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A3S2 AUROC: %f"%A3S2NovelAUROC)

# A4S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S1NovelFPR, A4S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A4S1 AUROC: %f"%A4S1NovelAUROC)

# A4S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S2NovelFPR, A4S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A4S2 AUROC: %f"%A4S2NovelAUROC)

# A5S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S1NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S1NovelFPR, A5S1NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A5S1 AUROC: %f"%A5S1NovelAUROC)

# A5S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S2NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S2NovelFPR, A5S2NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A5S2 AUROC: %f"%A5S2NovelAUROC)

# A6S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S1NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S1NovelFPR, A6S1NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A6S1 AUROC: %f"%A6S1NovelAUROC)

# A6S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S2NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S2NovelFPR, A6S2NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A6S2 AUROC: %f"%A6S2NovelAUROC)

#####################################################################
# Generate AUROC Graph (Method 1)
print("\nGenerating AUROC graph for Method 2\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Method_2_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Method_2_AUROC/")

plt.title("ROC Curve Novel Class Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(A1S1NovelFPR, A1S1NovelTPR, label = "A1S1")
plt.plot(A1S2NovelFPR, A1S2NovelTPR, label = "A1S2")
plt.plot(A2S1NovelFPR, A2S1NovelTPR, label = "A2S1")
plt.plot(A2S2NovelFPR, A2S2NovelTPR, label = "A2S2")
plt.plot(A3S1NovelFPR, A3S1NovelTPR, label = "A3S1")
plt.plot(A3S2NovelFPR, A3S2NovelTPR, label = "A3S2")
plt.plot(A4S1NovelFPR, A4S1NovelTPR, label = "A4S1")
plt.plot(A4S2NovelFPR, A4S2NovelTPR, label = "A4S2")
plt.plot(A5S1NovelFPR, A5S1NovelTPR, label = "A5S1")
plt.plot(A5S2NovelFPR, A5S2NovelTPR, label = "A5S2")
plt.plot(A6S1NovelFPR, A6S1NovelTPR, label = "A6S1")
plt.plot(A6S2NovelFPR, A6S2NovelTPR, label = "A6S2")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train1/Method_2_AUROC/method_2_AUROC.png")
plt.clf()

#####################################################################
# Saving data to CSV
csv_header = [
              " ",
              " ",
              "A1S1 Accuracy", 
              "A1S2 Accuracy", 
              "A2S1 Accuracy", 
              "A2S2 Accuracy", 
              "A3S1 Accuracy", 
              "A3S2 Accuracy", 
              "A4S1 Accuracy", 
              "A4S2 Accuracy", 
              "A5S1 Accuracy", 
              "A5S2 Accuracy", 
              "A6S1 Accuracy", 
              "A6S2 Accuracy", 

              "A1S1 AUROC", 
              "A1S2 AUROC", 
              "A2S1 AUROC", 
              "A2S2 AUROC", 
              "A3S1 AUROC", 
              "A3S2 AUROC", 
              "A4S1 AUROC", 
              "A4S2 AUROC", 
              "A5S1 AUROC", 
              "A5S2 AUROC"
              "A6S1 AUROC"
              "A6S2 AUROC"
            ]
outcsv = os.getcwd() + "/Results/FYP_compare.csv"

with open(outcsv, 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(csv_header)

insertData = []

rowsCsv = []

rowsCsv.append("Train 1")
rowsCsv.append("Method 1")
rowsCsv.append(A1S1Accuracy)  
rowsCsv.append(A1S2Accuracy)  
rowsCsv.append(A2S1Accuracy)  
rowsCsv.append(A2S2Accuracy)  
rowsCsv.append(A3S1Accuracy)  
rowsCsv.append(A3S2Accuracy)  
rowsCsv.append(A4S1Accuracy)  
rowsCsv.append(A4S2Accuracy)  
rowsCsv.append(A5S1Accuracy)  
rowsCsv.append(A5S2Accuracy)  
rowsCsv.append(A6S1Accuracy)  
rowsCsv.append(A6S2Accuracy)  

rowsCsv.append(A1S1AUROC)
rowsCsv.append(A1S2AUROC)
rowsCsv.append(A2S1AUROC)
rowsCsv.append(A2S2AUROC)
rowsCsv.append(A3S1AUROC)
rowsCsv.append(A3S2AUROC)
rowsCsv.append(A4S1AUROC)
rowsCsv.append(A4S2AUROC)
rowsCsv.append(A5S1AUROC)
rowsCsv.append(A5S2AUROC)
rowsCsv.append(A6S1AUROC)
rowsCsv.append(A6S2AUROC)

insertData.append(rowsCsv)
rowsCsv = []

rowsCsv.append("Train 1")
rowsCsv.append("Method 2")

rowsCsv.append(A1S1NovelAccuracy)  
rowsCsv.append(A1S2NovelAccuracy)  
rowsCsv.append(A2S1NovelAccuracy)  
rowsCsv.append(A2S2NovelAccuracy)  
rowsCsv.append(A3S1NovelAccuracy)  
rowsCsv.append(A3S2NovelAccuracy)  
rowsCsv.append(A4S1NovelAccuracy)  
rowsCsv.append(A4S2NovelAccuracy)  
rowsCsv.append(A5S1NovelAccuracy)  
rowsCsv.append(A5S2NovelAccuracy)  
rowsCsv.append(A6S1NovelAccuracy)  
rowsCsv.append(A6S2NovelAccuracy)  

rowsCsv.append(A1S1NovelAUROC)
rowsCsv.append(A1S2NovelAUROC)
rowsCsv.append(A2S1NovelAUROC)
rowsCsv.append(A2S2NovelAUROC)
rowsCsv.append(A3S1NovelAUROC)
rowsCsv.append(A3S2NovelAUROC)
rowsCsv.append(A4S1NovelAUROC)
rowsCsv.append(A4S2NovelAUROC)
rowsCsv.append(A5S1NovelAUROC)
rowsCsv.append(A5S2NovelAUROC)
rowsCsv.append(A6S1NovelAUROC)
rowsCsv.append(A6S2NovelAUROC)


insertData.append(rowsCsv)

with open(outcsv, 'a', newline='', encoding='UTF8') as fileCsv:
  writer = csv.writer(fileCsv)

  writer.writerows(insertData)

#####################################################################
# Redefining parameter for retraining 
print("\nRetraining autoencoder\n")
novelClassCollection = [0, 1, 2]
lengthNovelClassCollection = len(novelClassCollection)

actualNovelClass = [3, 4, 5]
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
#####################################################################

## Loading dataset
from keras.datasets import mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#####################################################################
# Processing dataset for later use

# Testing dataset for method 1
print("\nProcessing Testing dataset 1, contains novel and non-novel labels")
test_loader_no_actual, test_label_no_actual = processImage(test_X, 
                                       test_Y, 
                                       removeNovel = False,
                                       removeActualNovel = True, 
                                       novelClassCollection = novelClassCollection,
                                       actualNovelClass = actualNovelClass,
                                       classToUse = classToUse
                                       )

# Testing dataset, contains all labels for method 2
print("\nProcessing Testing dataset 2, contains all labels, ie actual, novel and non-novel labels")
test_loader, test_label = processImage(test_X, 
                                       test_Y, 
                                       removeNovel = False,
                                       removeActualNovel = False, 
                                       novelClassCollection = novelClassCollection,
                                       actualNovelClass = actualNovelClass,
                                       classToUse = classToUse
                                       )


# Training dataset for autoencoder, contains non-novel label only
print("\nProcessing Train dataset 1, contains non-novel labels")
trainData, trainLabel = processImage(train_X, 
                                     train_Y, 
                                     removeNovel = True,
                                     removeActualNovel = True,
                                     novelClassCollection = novelClassCollection,
                                     actualNovelClass = actualNovelClass,
                                     classToUse = classToUse
                                    )

# Training dataset for Siamese Network, contains novel label and non-novel label. Does not have actual novel label
print("\nProcessing Train dataset 2, contains novel and non-novel labels")
trainOtherData, trainOtherLabel = processImage(train_X, 
                                               train_Y, 
                                               removeNovel = False,
                                               removeActualNovel = True,
                                               novelClassCollection = novelClassCollection,
                                               actualNovelClass = actualNovelClass,
                                               classToUse = classToUse
                                            )

#####################################################################
# Convert data into tensor
# Test Data for method 2
test_loader = torch.tensor(test_loader)
test_label = torch.tensor(test_label)

# Testing dataset for method 1
test_loader_no_actual = torch.tensor(test_loader_no_actual)
test_label_no_actual = torch.tensor(test_label_no_actual)

# Train Data for Autoencoder
trainData = torch.tensor(trainData)
trainLabel = torch.tensor(trainLabel)

# Train Data for Siamese Network
trainOtherData = torch.tensor(trainOtherData)
trainOtherLabel = torch.tensor(trainOtherLabel)

#####################################################################
# Convert tensor into tensor dataset

# Train data for Autoencoder
trainData2 = TensorDataset(trainData, trainLabel)

# Train data for Siamese Network
trainOtherData = TensorDataset(trainOtherData, trainOtherLabel)

# Test Data for method 2
test_loader = TensorDataset(test_loader, test_label)

# Test data for method 1
test_loader_no_actual= TensorDataset(test_loader_no_actual, test_label_no_actual)

#####################################################################
# Convert tensor dataset into Dataloader, make them into batch sizes, and shuffle

# Train data for Autoencoder
train_loader = DataLoader(trainData2, batch_size, shuffle=True)

# Train data for Siamese Network
train_other_loader = DataLoader(trainOtherData, batch_size, shuffle=True)

#####################################################################
# make dataloader to device

# Train data for Autoencoder
train_loader = DeviceDataLoader(train_loader, device)

# Train data for Siamese Network
train_other_loader = DeviceDataLoader(train_other_loader, device)

# Test data with actual novel label, novel label and non-novel label - method 2
test_loader = DeviceDataLoader(test_loader, device)

# Test data with novel label and non-novel label - method 1
test_loader_no_actual= DeviceDataLoader(test_loader_no_actual, device)

#####################################################################
# get knnImage and knnLabel
print("\nPreparing dataset for KNN for Variant 4 Autoencoder\n")
knnImage = []
oriImage = []

for data in trainData2:
  
  img, label  = data

  imageTemp = img.to("cpu").numpy()
  
  flattenedShape = imageTemp.shape[1] * imageTemp.shape[2]
  imageTemp = np.reshape(imageTemp, (flattenedShape))

  knnImage.append(imageTemp.tolist())
  oriImage.append(img)

#####################################################################
# Define and Train the first KNN Model
numberOfClusters = 4


print("\nTraining KNN For Variant 4 Autoencoder\n")

## as of now the number of clusters depend highly on the number of labels used

knnModel = KMeans(n_clusters=numberOfClusters)

knnModel.fit(knnImage)


#####################################################################
# Define function to get image from KNN 1 to train Autoencoder
counter = 0
autoencoderTrainImage = []
for index in range(numberOfClusters):
  while True:
    knnImageIndex = knnImage[counter]
    oriImageIndex = oriImage[counter]

    predictedValue = knnModel.predict([knnImageIndex])
    if(predictedValue == index):
      autoencoderTrainImage.append(oriImageIndex.to("cpu").numpy())
      break

    counter = counter + 1

#####################################################################
# Define function to get a single image from test image dataset
imageCollectionForAllIndex = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
}
imageCollectionCountForAllIndex = [
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100
]
for image, label in trainOtherData:
  labelCount = imageCollectionCountForAllIndex[label.item()]
  if(labelCount == 0):
    continue
  else:
    # there is still count left
    imageCollectionCountForAllIndex[label.item()] = imageCollectionCountForAllIndex[label.item()] - 1

    arrayIndex = imageCollectionForAllIndex[label.item()]

    arrayIndex.append(image)

    imageCollectionForAllIndex[label.item()] = arrayIndex
  

imageCollectionForIndex = []
otherClassLength = len(train_other_loaderClass)
for index in range(otherClassLength):
  index = train_other_loaderClass[index]
  tempImage = getImageLabelTrain(index,1, imageCollectionForAllIndex)
  # invert the image
  # tempImageOnes = torch.tensor(np.ones(tempImage.shape))
  # tempImage = tempImageOnes - tempImage
  imageCollectionForIndex.append(tempImage.to("cpu").numpy())

#####################################################################
# Define the forced encoder representation
print("\nDefining Forced Encoder Representation for Variant 2 and Variant 4 Autoencoder\n")
encoderLatentCollection = []
index = 0

while index  < (numberOfClusters):
  tempRandomNumberCollection = (np.random.normal(0,1,latent_space_features).tolist())   
  if(check_distance_between_latent(tempRandomNumberCollection, encoderLatentCollection, latent_space_features)):
    print("1 Completed")
    encoderLatentCollection.append(tempRandomNumberCollection)
    index = index + 1
  else:
    continue


#####################################################################
# Define Encoder and Decoder
## Autoencoder for KNN training for Siamese Network
print("\nDeclaring Autoencoders\n")
encoderNet0 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp0 = torch.optim.Adam(encoderNet0.parameters(), lr=learning_rate)

decoderNet0 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp0 = torch.optim.Adam(decoderNet0.parameters(), lr=learning_rate)

## Autoencoder for evaluating performance, trained in normal fashion 
encoderNet = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp = torch.optim.Adam(encoderNet.parameters(), lr=learning_rate)

decoderNet = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp = torch.optim.Adam(decoderNet.parameters(), lr=learning_rate)

## KNN Force Autoencoder Collection  
encoderNet2 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp2 = torch.optim.Adam(encoderNet2.parameters(), lr=learning_rate)

decoderNet2 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp2 = torch.optim.Adam(decoderNet2.parameters(), lr=learning_rate)

encoderNet3 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp3 = torch.optim.Adam(encoderNet3.parameters(), lr=learning_rate)

decoderNet3 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp3 = torch.optim.Adam(decoderNet3.parameters(), lr=learning_rate)

encoderNet4 = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp4 = torch.optim.Adam(encoderNet4.parameters(), lr=learning_rate)

decoderNet4 = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp4 = torch.optim.Adam(decoderNet4.parameters(), lr=learning_rate)

# Supervised Force Autoencoder COllection
encoderNet_supervised = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp_supervised = torch.optim.Adam(encoderNet_supervised.parameters(), lr=learning_rate)

decoderNet_supervised = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp_supervised = torch.optim.Adam(decoderNet_supervised.parameters(), lr=learning_rate)

encoderNet2_supervised = encoder(number_of_conv_final_channel, latent_space_features, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
encoderOp2_supervised = torch.optim.Adam(encoderNet2_supervised.parameters(), lr=learning_rate)

decoderNet2_supervised = decoder(latent_space_features, number_of_conv_final_channel, conv_image_size, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
decoderOp2_supervised = torch.optim.Adam(decoderNet2_supervised.parameters(), lr=learning_rate)


#####################################################################
# Loading/Training Autoencoder for KNN 
if(os.path.exists(os.getcwd() + "/Data/retrain_encoderTrainedKNN.pth") and os.path.exists(os.getcwd() + "/Data/retrain_decoderTrainedKNN.pth")):
  print("\nLoading encoder and decoder for KNN for variant 5 Autoencoder\n")
  encoderNet0.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_encoderTrainedKNN.pth", map_location=device))
  decoderNet0.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_decoderTrainedKNN.pth", map_location=device))
else:
  print("\nTraining encoder and decoder for KNN for variant 5 Autoencoder\n")
  num_epochs = 40
  retrainDecoderTimes = 100


  for epoch in tqdm(range(0, num_epochs), desc ="Training Autoencoder for KNN"):

    for idx, data in enumerate(train_loader, 0):
        # new line
        # set the model into a train state
        encoderNet0.train()
        decoderNet0.train()
      
        imgs, _ = data

        imgs = imgs.float()

      
        outEncoder = encoderNet0(imgs)
        outDecoder = decoderNet0(outEncoder)
        lossAutoencoder = nn.functional.mse_loss(outDecoder, imgs)
        
        encoderOp0.zero_grad()
        decoderOp0.zero_grad()
        lossAutoencoder.backward()
        encoderOp0.step()
        decoderOp0.step()
        

    
    print('\nEpoch {}: Autoencoder Loss {}'.format(epoch, lossAutoencoder))   

  # saving model
  torch.save(encoderOp0.state_dict(), os.getcwd() + "/Data/encoderTrainedKNN.pth")
  torch.save(decoderOp0.state_dict(), os.getcwd() + "/Data/decoderTrainedKNN.pth")

#####################################################################
# Train KNN 3 to get representation from encoder
outputEncoderCollectionKNN = []

for data in trainData2:
  
  img, label  = data

  img = img.unsqueeze(0).to(device).float()

  outputEncoder = encoderNet0(img)
  
  outputEncoderCollectionKNN.append(outputEncoder.detach().to("cpu").squeeze(0).numpy().tolist())

numberOfClusters3 = 4

print("\nTraining KNN for variant 5 Autoencoder using latent space from normal autoencoder\n")
knnModel3 = KMeans(n_clusters=numberOfClusters3)

knnModel3.fit(outputEncoderCollectionKNN)

#####################################################################
# Preparing dataset for training of autoencoder variants..
print("\nPreparing dataset for training of autoencoder variants..\n")
autoencoderEncoderCollection = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[]
}
for data in trainData2:
  
  img, label  = data

  img = img.unsqueeze(0).to(device).float()

  outputEncoder = encoderNet0(img.float())

  outputEncoder = outputEncoder.detach().to("cpu").squeeze(0).numpy().tolist()

  predictedValue = knnModel3.predict([outputEncoder])

  autoencoderEncoderCollection[predictedValue[0]].append(outputEncoder)

#####################################################################
# Generating encoder collection for Variant 5 Autoencoder

autoencoderEncoderCollectionAvg = []
for index in range(10): 
  encoderAutoencoderCollectionIndex = torch.tensor(autoencoderEncoderCollection[index])
  averageEncoder = torch.mean(encoderAutoencoderCollectionIndex, axis = 0)
  # print(averageEncoder.shape)
  if(averageEncoder.shape == torch.tensor(float("nan")).shape):
    q=1
  else:
    autoencoderEncoderCollectionAvg.append(averageEncoder.squeeze(0).numpy().tolist())


#####################################################################
# Generating train image for Variant 5 Autoencoder
counter = 0
autoencoderTrainImage2 = []
for index in range(numberOfClusters3):
  while True:
    oriImageIndex = oriImage[counter]

    oriImageIndex = oriImageIndex.unsqueeze(0).to(device).float()
    outputEncoder = encoderNet0(oriImageIndex)

    outputEncoder = outputEncoder.detach().to("cpu").float().numpy().astype(np.float64)


    predictedLabel = knnModel3.predict(outputEncoder)

    if(predictedLabel[0] == index):
      autoencoderTrainImage2.append(oriImageIndex.to("cpu").numpy())
      break


    counter = counter + 1

#####################################################################
# Get average latent space of autoencoder for training Variant 3 autoencoder
encoderAutoencoderCollection = {
    0:[],
    1:[],
    2:[],
    3:[],
    4:[],
    5:[],
    6:[],
    7:[],
    8:[],
    9:[],
}
encoderAutoencoderRepresentation = []
for data in trainData2:
  encoderNet0.eval()
  decoderNet0.eval()
  
  img, label = data
  img = img.unsqueeze(0).float().to(device)

  outEncoder = encoderNet0(img)
      
  encoderAutoencoderCollection[label.item()].append(outEncoder.detach().to("cpu").numpy().tolist())

for index in range(10):
  encoderAutoencoderCollectionIndex = torch.tensor(encoderAutoencoderCollection[index])
  averageEncoder = torch.mean(encoderAutoencoderCollectionIndex, axis = 0)
  # print(averageEncoder.shape)
  if(averageEncoder.shape == torch.tensor(float("nan")).shape):
    q=1
  else:
    encoderAutoencoderRepresentation.append(averageEncoder.squeeze(0).numpy().tolist())

#####################################################################
# Training autoencoder
if(os.path.exists(os.getcwd() + "/Data/retrain_normalEncoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/retrain_normalDecoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/retrain_variant4Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/retrain_variant4Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/retrain_variant5Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/retrain_variant5Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/retrain_variant6Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/retrain_variant6Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/retrain_variant2Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/retrain_variant2Decoder.pth") and 

    os.path.exists(os.getcwd() + "/Data/retrain_variant3Encoder.pth") and 
    os.path.exists(os.getcwd() + "/Data/retrain_variant3Decoder.pth")
    ):
  print("\nLoading Autoencoder\n")
  encoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_normalEncoder.pth", map_location=device))
  decoderNet.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_normalDecoder.pth", map_location=device))

  encoderNet2.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant4Encoder.pth", map_location=device))
  decoderNet2.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant4Decoder.pth", map_location=device))

  encoderNet3.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant5Encoder.pth", map_location=device))
  decoderNet3.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant5Decoder.pth", map_location=device))

  encoderNet4.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant6Encoder.pth", map_location=device))
  decoderNet4.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant6Decoder.pth", map_location=device))

  encoderNet_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant2Encoder.pth", map_location=device))
  decoderNet_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant2Decoder.pth", map_location=device))

  encoderNet2_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant3Encoder.pth", map_location=device))
  decoderNet2_supervised.load_state_dict(torch.load(os.getcwd() + "/Data/retrain_variant3Decoder.pth", map_location=device))

else:
  print("\nTraining Autoencoder\n")
  num_epochs = 40
  retrainDecoderTimes = 100


  for epoch in tqdm(range(0, num_epochs), desc ="Training Autoencoder"):

    for idx, data in enumerate(train_loader, 0):
        # new line
        # set the model into a train state
        encoderNet.train()
        decoderNet.train()

        encoderNet2.train()
        decoderNet2.train()

        encoderNet3.train()
        decoderNet3.train()

        encoderNet4.train()
        decoderNet4.train()

        encoderNet_supervised.train()
        decoderNet_supervised.train()

        encoderNet2_supervised.train()
        decoderNet2_supervised.train()


        imgs, actualLabel = data


        # KNN Variant 4 - perform KNN in the image to get label for images
        flattenedImage = flattenKNNCollection(imgs)

        label = knnModel.predict(flattenedImage)

        # get the latent representation for the associated label   
        encoderRepresentation = getEncoderLatentCollection(label, encoderLatentCollection, device)
        
        imgs = imgs.float()

        imagesCompare = getImageForEncoderCollection(label, autoencoderTrainImage, device).float()
      
        outEncoder = encoderNet2(imgs)

        lossEncoder_variant4 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp2.zero_grad()
        lossEncoder_variant4.backward()
        encoderOp2.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet2(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet2(outEncoder)
        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant4 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp2.zero_grad()
        lossDecoder_variant4.backward()
        decoderOp2.step()
        
        # Train the autoencoder Variant 1
        outEncoder = encoderNet(imgs)
        outDecoder = decoderNet(outEncoder)

        lossAutoencoder = nn.functional.mse_loss(outDecoder, imgs)

        encoderOp.zero_grad()
        decoderOp.zero_grad()
        lossAutoencoder.backward()
        encoderOp.step()
        decoderOp.step()


        # Train the KNN Encoder Autoencoder Variant 5 - use latent space to KNN to produce label train autoencoder
        labelsCollection = generateLabelRepresentation(imgs, encoderNet0, knnModel3)
        
        encoderRepresentation = getAutoencoderEncoderRepresentation(labelsCollection, autoencoderEncoderCollectionAvg, device)

        imageCollection = getImageForEncoderCollection2(labelsCollection, autoencoderTrainImage2, device)

        outEncoder = encoderNet3(imgs)

        lossEncoder_variant5 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp3.zero_grad()
        lossEncoder_variant5.backward()
        encoderOp3.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet3(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet3(outEncoder)
        imagesCompare = ((0.9 * imageCollection) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant5 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp3.zero_grad()
        lossDecoder_variant5.backward()
        decoderOp3.step()

        # Train the new autoencoder variant 6
        labelsCollection = generateLabelRepresentation(imgs, encoderNet0, knnModel3)
        
        encoderRepresentation = getEncoderLatentCollection(labelsCollection, encoderLatentCollection, device)  # change here

        imageCollection = getImageForEncoderCollection2(labelsCollection, autoencoderTrainImage2, device)

        outEncoder = encoderNet4(imgs)

        lossEncoder_variant6 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp4.zero_grad()
        lossEncoder_variant6.backward()
        encoderOp4.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet4(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet4(outEncoder)
        imagesCompare = ((0.9 * imageCollection) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant6 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp4.zero_grad()
        lossDecoder_variant6.backward()
        decoderOp4.step()


        ############ train supervised autoencoder
        # variant 2
        # get the random latent representation for the associated label
        encoderRepresentation = getEncoderLatentCollection_supervised(actualLabel, familiarClass, encoderLatentCollection, device)
        imagesCompare = getImageCollectionLabelTrain(actualLabel, train_other_loaderClass, imageCollectionForIndex, device).float()

        outEncoder = encoderNet_supervised(imgs)

        lossEncoder_variant2 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp_supervised.zero_grad()
        lossEncoder_variant2.backward()
        encoderOp_supervised.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet_supervised(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet_supervised(outEncoder)
        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant2 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp_supervised.zero_grad()
        lossDecoder_variant2.backward()
        decoderOp_supervised.step()


        # Train the autoencoder variant 3 - use latent space from autoencoder averaged
        encoderRepresentation = getAutoencoderEncoderRepresentation_supervised(actualLabel, familiarClass, encoderAutoencoderRepresentation, device)

        outEncoder = encoderNet2_supervised(imgs)
        lossEncoder_variant3 = nn.functional.mse_loss(outEncoder, encoderRepresentation)

        encoderOp2_supervised.zero_grad()
        lossEncoder_variant3.backward()
        encoderOp2_supervised.step()


        outEncoder = encoderNet2_supervised(imgs)
        outEncoder = outEncoder.detach()
        outDecoder = decoderNet2_supervised(outEncoder)

        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant3 = nn.functional.mse_loss(outDecoder, imagesCompare)

        decoderOp2_supervised.zero_grad()
        lossDecoder_variant3.backward()
        decoderOp2_supervised.step()



    
    print('\nEpoch {}: Variant 1 Loss {}'.format(epoch, lossAutoencoder))   
    print('Epoch {}: Variant 2 Encoder Loss {}'.format(epoch, lossEncoder_variant2))
    print('Epoch {}: Variant 2 Decoder Loss {}'.format(epoch, lossDecoder_variant2))

    print('Epoch {}: Variant 3 Encoder Loss {}'.format(epoch, lossEncoder_variant3))
    print('Epoch {}: Variant 3 Decoder Loss {}'.format(epoch, lossDecoder_variant3))

    print('Epoch {}: Variant 4 Encoder Loss {}'.format(epoch, lossEncoder_variant4))
    print('Epoch {}: Variant 4 Decoder Loss {}'.format(epoch, lossDecoder_variant4))

    print('Epoch {}: Variant 5 Encoder Loss {}'.format(epoch, lossEncoder_variant5))
    print('Epoch {}: Variant 5 Decoder Loss {}'.format(epoch, lossDecoder_variant5))

    print('Epoch {}: Variant 6 Encoder Loss {}'.format(epoch, lossEncoder_variant6))
    print('Epoch {}: Variant 6 Decoder Loss {}'.format(epoch, lossDecoder_variant6))
  
  # saving model
  torch.save(encoderNet.state_dict(), os.getcwd() + "/Data/retrain_normalEncoder.pth")
  torch.save(decoderNet.state_dict(), os.getcwd() + "/Data/retrain_normalDecoder.pth")

  torch.save(encoderNet2.state_dict(), os.getcwd() + "/Data/retrain_variant4Encoder.pth")
  torch.save(decoderNet2.state_dict(), os.getcwd() + "/Data/retrain_variant4Decoder.pth")

  torch.save(encoderNet3.state_dict(), os.getcwd() + "/Data/retrain_variant5Encoder.pth")
  torch.save(decoderNet3.state_dict(), os.getcwd() + "/Data/retrain_variant5Decoder.pth")

  torch.save(encoderNet4.state_dict(), os.getcwd() + "/Data/retrain_variant6Encoder.pth")
  torch.save(decoderNet4.state_dict(), os.getcwd() + "/Data/retrain_variant6Decoder.pth")

  torch.save(encoderNet_supervised.state_dict(), os.getcwd() + "/Data/retrain_variant2Encoder.pth")
  torch.save(decoderNet_supervised.state_dict(), os.getcwd() + "/Data/retrain_variant2Decoder.pth")

  torch.save(encoderNet2_supervised.state_dict(), os.getcwd() + "/Data/retrain_variant3Encoder.pth")
  torch.save(decoderNet2_supervised.state_dict(), os.getcwd() + "/Data/retrain_variant3Decoder.pth")

#####################################################################
# Save Variant 1Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 1 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "1"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet.eval()
      decoderNet.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet(outputModelIn)
      outputModel = decoderNet(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 2 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 2 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "2"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet_supervised.eval()
      decoderNet_supervised.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet_supervised(outputModelIn)
      outputModel = decoderNet_supervised(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 3 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 3 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "3"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet2_supervised .eval()
      decoderNet2_supervised .eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet2_supervised (outputModelIn)
      outputModel = decoderNet2_supervised (outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 4 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 4 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "4"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet2.eval()
      decoderNet2.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet2(outputModelIn)
      outputModel = decoderNet2(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 5 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 5 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "5"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet3.eval()
      decoderNet3.eval()
          

      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet3(outputModelIn)
      outputModel = decoderNet3(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Save Variant 6 Autoencoder Input and Output Image in directory for comparision
print("\nSave Variant 6 Autoencoder Input and Output Image in directory for comparision\n")
imageLoopStart = 10
numberOfCompareImage = 10
autoencoderVariant = "6"
finalImageCollection = np.array([])
for index in range(10):
  imageLabel = index
  finalImage = np.array([])
  with torch.no_grad():
    for imageCompareIndex in range(numberOfCompareImage):

      image= getImageLabel2(imageLabel, imageCompareIndex + imageLoopStart, test_loader).to(device)


      encoderNet4.eval()
      decoderNet4.eval()
          
      outputModelIn = image.unsqueeze(0).float().to(device)

      outputEncoderModel = encoderNet4(outputModelIn)
      outputModel = decoderNet4(outputEncoderModel)

      outputModelShow = outputModel.squeeze(0)
      imageCompare = image.squeeze(0)

      producedImage = torch.cat((image.to("cpu"), outputModelShow.to("cpu")),2)

      producedImage = producedImage.to("cpu").detach().squeeze(0).squeeze(0).numpy()  

      if(finalImage.shape == np.array([]).shape):
        finalImage = producedImage
      else:
        finalImage = cv2.vconcat([finalImage, producedImage])
      paddingImage = np.zeros((30, producedImage.shape[1]))
      finalImage = cv2.vconcat([finalImage, paddingImage])

    if(finalImageCollection.shape == np.array([]).shape):
      finalImageCollection = finalImage
    else:
      finalImageCollection = cv2.hconcat([finalImageCollection, finalImage])
    paddingImage = np.ones((finalImageCollection.shape[0], 20))
    finalImageCollection = cv2.hconcat([finalImageCollection, paddingImage])
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/")
cv2.imwrite(os.getcwd() + "/Results/Train2/Variant_"+autoencoderVariant+"_Autoencoder/results.png", finalImageCollection*255)

#####################################################################
# Check accuracy of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# A1S1
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A1S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A1S1 Accuracy: %f"%A1S1Accuracy)


# A1S2
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A1S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A1S2 Accuracy: %f"%A1S2Accuracy)

# A2S1
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)  

    imageIn = decoderNet_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A2S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A2S1 Accuracy: %f"%A2S1Accuracy)

# A2S2
with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)

    imageIn = decoderNet_supervised(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A2S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A2S2 Accuracy: %f"%A2S2Accuracy)

# A3S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A3S1 Accuracy: %f"%A3S1Accuracy)

# A3S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A3S2 Accuracy: %f"%A3S2Accuracy)

# A4S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A4S1 Accuracy: %f"%A4S1Accuracy)


# A4S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A4S2 Accuracy: %f"%A4S2Accuracy)


# A5S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A5S1 Accuracy: %f"%A5S1Accuracy)


# A5S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A5S2 Accuracy: %f"%A5S2Accuracy)

# A6S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A6S1Accuracy = correctElements/totalElements * 100

  print("Method 1 A6S1 Accuracy: %f"%A6S1Accuracy)


# A6S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A6S2Accuracy = correctElements/totalElements * 100

  print("Method 1 A6S2 Accuracy: %f"%A6S2Accuracy)

#####################################################################
# Check AUROC of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# A1S1

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S1FPR, A1S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)
print("Method 1 A1S1 AUROC: %f"%A1S1AUROC)

# A1S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S2FPR, A1S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A1S2 AUROC: %f"%A1S2AUROC)

# A2S1

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S1FPR, A2S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A2S1 AUROC: %f"%A2S1AUROC)

# A2S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet_supervised.eval()
    decoderNet_supervised.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S2FPR, A2S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A2S2 AUROC: %f"%A2S2AUROC)

# A3S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S1FPR, A3S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A3S1 AUROC: %f"%A3S1AUROC)

# A3S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S2FPR, A3S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A3S2 AUROC: %f"%A3S2AUROC)

# A4S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S1FPR, A4S1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A4S1 AUROC: %f"%A4S1AUROC)

# A4S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S2FPR, A4S2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 A4S2 AUROC: %f"%A4S2AUROC)

# A5S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S1AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S1FPR, A5S1TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A5S1 AUROC: %f"%A5S1AUROC)

# A5S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S2AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S2FPR, A5S2TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A5S2 AUROC: %f"%A5S2AUROC)

# A6S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S1AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S1FPR, A6S1TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A6S1 AUROC: %f"%A6S1AUROC)

# A6S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S2AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S2FPR, A6S2TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 A6S2 AUROC: %f"%A6S2AUROC)

#####################################################################
# Generate AUROC graph
print("\nGenerating AUROC graph for Method 1\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Method_1_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Method_1_AUROC/")

plt.title("ROC Curve Actual Novel Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(A1S1FPR, A1S1TPR, label = "A1S1")
plt.plot(A1S2FPR, A1S2TPR, label = "A1S2")
plt.plot(A2S1FPR, A2S1TPR, label = "A2S1")
plt.plot(A2S2FPR, A2S2TPR, label = "A2S2")
plt.plot(A3S1FPR, A3S1TPR, label = "A3S1")
plt.plot(A3S2FPR, A3S2TPR, label = "A3S2")
plt.plot(A4S1FPR, A4S1TPR, label = "A4S1")
plt.plot(A4S2FPR, A4S2TPR, label = "A4S2")
plt.plot(A5S1FPR, A5S1TPR, label = "A5S1")
plt.plot(A5S2FPR, A5S2TPR, label = "A5S2")
plt.plot(A6S1FPR, A6S1TPR, label = "A6S1")
plt.plot(A6S2FPR, A6S2TPR, label = "A6S2")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train2/Method_1_AUROC/method_1_AUROC.png")
plt.clf()

#####################################################################
# Check accuracy of variants (Method 2)
# Method 2 consist of novel class and non-novel class only. It does not contain actual novel class
# The label used for testing is similar with the label used to train Siamese Network
# The data are different, with testing and training dataset

# A1S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A1S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A1S1 Accuracy: %f"%A1S1NovelAccuracy)

# A1S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet .eval()
    decoderNet .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet (img)

    imageIn = decoderNet (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A1S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A1S2 Accuracy: %f"%A1S2NovelAccuracy)

# A2S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)  

    imageIn = decoderNet_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A2S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A2S1 Accuracy: %f"%A2S1NovelAccuracy)

# A2S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)

    imageIn = decoderNet_supervised(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A2S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A2S2 Accuracy: %f"%A2S2NovelAccuracy)

# A3S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A3S1 Accuracy: %f"%A3S1NovelAccuracy)

# A3S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)

    imageIn = decoderNet2_supervised (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A3S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A3S2 Accuracy: %f"%A3S2NovelAccuracy)

# A4S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A4S1 Accuracy: %f"%A4S1NovelAccuracy)

# A4S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)

    imageIn = decoderNet2 (outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A4S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A4S2 Accuracy: %f"%A4S2NovelAccuracy)

# A5S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A5S1 Accuracy: %f"%A5S1NovelAccuracy)

# A5S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)

    imageIn = decoderNet3(outputEncoder)
    
    out1, out2 = novelKNN(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1



  A5S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A5S2 Accuracy: %f"%A5S2NovelAccuracy)

# A6S1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A6S1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A6S1 Accuracy: %f"%A6S1NovelAccuracy)


# A6S2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)

    imageIn = decoderNet4(outputEncoder)
    
    out1, out2 = novelSupervised(img,imageIn)

    distance = F.pairwise_distance(out1, out2)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    if(checkNovelAccuracy(distance,novelLabel)):
      correctElements = correctElements + 1

    totalElements = totalElements + 1

  A6S2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 A6S2 Accuracy: %f"%A6S2NovelAccuracy)

#####################################################################
# Get AUROC for variants


# A1S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S1NovelFPR, A1S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A1S1 AUROC: %f"%A1S1NovelAUROC)


# A1S2

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet.eval()
    decoderNet.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A1S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A1S2NovelFPR, A1S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A1S2 AUROC: %f"%A1S2NovelAUROC)

# A2S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised .eval()
    decoderNet_supervised.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S1NovelFPR, A2S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 3 AUROC: %f"%A2S1NovelAUROC)


# A2S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet_supervised.eval()
    decoderNet_supervised.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet_supervised(img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet_supervised(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A2S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A2S2NovelFPR, A2S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A2S2 AUROC: %f"%A2S2NovelAUROC)

# A3S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S1NovelFPR, A3S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A3S1 AUROC: %f"%A3S1NovelAUROC)


# A3S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2_supervised .eval()
    decoderNet2_supervised .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2_supervised (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2_supervised (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A3S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A3S2NovelFPR, A3S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A3S2 AUROC: %f"%A3S2NovelAUROC)

# A4S1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S1NovelFPR, A4S1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A4S1 AUROC: %f"%A4S1NovelAUROC)

# A4S2
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet2 .eval()
    decoderNet2 .eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet2 (img)
    outputEncoder = outputEncoder.detach()

    outDecoder = decoderNet2 (outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    outputNovelLabel = outputNovelLabel.to("cpu").numpy()

    outputNovelLabel = outputNovelLabel.tolist()


    noveltyActualNovel.append(novelLabel[0])

    noveltyPredictedNovel.append(outputNovelLabel[0])

A4S2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
A4S2NovelFPR, A4S2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 A4S2 AUROC: %f"%A4S2NovelAUROC)

# A5S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S1NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S1NovelFPR, A5S1NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A5S1 AUROC: %f"%A5S1NovelAUROC)

# A5S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet3.eval()
    decoderNet3.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet3(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet3(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A5S2NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A5S2NovelFPR, A5S2NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A5S2 AUROC: %f"%A5S2NovelAUROC)

# A6S1
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelSupervised.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelSupervised(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S1NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S1NovelFPR, A6S1NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A6S1 AUROC: %f"%A6S1NovelAUROC)

# A6S2
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_loader_no_actual, 0):
    encoderNet4.eval()
    decoderNet4.eval()
    novelKNN.eval()

    img, label = data

    img = img.to(device).float().unsqueeze(0)

    outputEncoder = encoderNet4(img)
    outputEncoder = outputEncoder.detach()

    outputDecoder = decoderNet4(outputEncoder)

    novelLabel = genNovelOrNotOnLabel([label], actualNovelClass, novelClassCollection)

    out1, out2 = novelKNN(img,outputDecoder)

    outputNovelLabel = F.pairwise_distance(out1, out2)

    finalModelNovel = outputNovelLabel.to("cpu").numpy()

    finalModelNovel = finalModelNovel.tolist()


    forcedNoveltyActualLabel.append(novelLabel[0])

    forcedNoveltyPredictedLabel.append(finalModelNovel[0])

    
A6S2NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
A6S2NovelFPR, A6S2NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 A6S2 AUROC: %f"%A6S2NovelAUROC)

#####################################################################
# Generate AUROC Graph (Method 1)
print("\nGenerating AUROC graph for Method 2\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Method_2_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Method_2_AUROC/")

plt.title("ROC Curve Novel Class Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(A1S1NovelFPR, A1S1NovelTPR, label = "A1S1")
plt.plot(A1S2NovelFPR, A1S2NovelTPR, label = "A1S2")
plt.plot(A2S1NovelFPR, A2S1NovelTPR, label = "A2S1")
plt.plot(A2S2NovelFPR, A2S2NovelTPR, label = "A2S2")
plt.plot(A3S1NovelFPR, A3S1NovelTPR, label = "A3S1")
plt.plot(A3S2NovelFPR, A3S2NovelTPR, label = "A3S2")
plt.plot(A4S1NovelFPR, A4S1NovelTPR, label = "A4S1")
plt.plot(A4S2NovelFPR, A4S2NovelTPR, label = "A4S2")
plt.plot(A5S1NovelFPR, A5S1NovelTPR, label = "A5S1")
plt.plot(A5S2NovelFPR, A5S2NovelTPR, label = "A5S2")
plt.plot(A6S1NovelFPR, A6S1NovelTPR, label = "A6S1")
plt.plot(A6S2NovelFPR, A6S2NovelTPR, label = "A6S2")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train2/Method_2_AUROC/method_2_AUROC.png")
plt.clf()

#####################################################################
# Saving data to CSV
outcsv = os.getcwd() + "/Results/FYP_compare.csv"

insertData = []

rowsCsv = []

rowsCsv.append("Train 2")
rowsCsv.append("Method 1")
rowsCsv.append(A1S1Accuracy)  
rowsCsv.append(A1S2Accuracy)  
rowsCsv.append(A2S1Accuracy)  
rowsCsv.append(A2S2Accuracy)  
rowsCsv.append(A3S1Accuracy)  
rowsCsv.append(A3S2Accuracy)  
rowsCsv.append(A4S1Accuracy)  
rowsCsv.append(A4S2Accuracy)  
rowsCsv.append(A5S1Accuracy)  
rowsCsv.append(A5S2Accuracy)  
rowsCsv.append(A6S1Accuracy)  
rowsCsv.append(A6S2Accuracy)  

rowsCsv.append(A1S1AUROC)
rowsCsv.append(A1S2AUROC)
rowsCsv.append(A2S1AUROC)
rowsCsv.append(A2S2AUROC)
rowsCsv.append(A3S1AUROC)
rowsCsv.append(A3S2AUROC)
rowsCsv.append(A4S1AUROC)
rowsCsv.append(A4S2AUROC)
rowsCsv.append(A5S1AUROC)
rowsCsv.append(A5S2AUROC)
rowsCsv.append(A6S1AUROC)
rowsCsv.append(A6S2AUROC)

insertData.append(rowsCsv)
rowsCsv = []

rowsCsv.append("Train 2")
rowsCsv.append("Method 2")

rowsCsv.append(A1S1NovelAccuracy)  
rowsCsv.append(A1S2NovelAccuracy)  
rowsCsv.append(A2S1NovelAccuracy)  
rowsCsv.append(A2S2NovelAccuracy)  
rowsCsv.append(A3S1NovelAccuracy)  
rowsCsv.append(A3S2NovelAccuracy)  
rowsCsv.append(A4S1NovelAccuracy)  
rowsCsv.append(A4S2NovelAccuracy)  
rowsCsv.append(A5S1NovelAccuracy)  
rowsCsv.append(A5S2NovelAccuracy)  
rowsCsv.append(A6S1NovelAccuracy)  
rowsCsv.append(A6S2NovelAccuracy)  

rowsCsv.append(A1S1NovelAUROC)
rowsCsv.append(A1S2NovelAUROC)
rowsCsv.append(A2S1NovelAUROC)
rowsCsv.append(A2S2NovelAUROC)
rowsCsv.append(A3S1NovelAUROC)
rowsCsv.append(A3S2NovelAUROC)
rowsCsv.append(A4S1NovelAUROC)
rowsCsv.append(A4S2NovelAUROC)
rowsCsv.append(A5S1NovelAUROC)
rowsCsv.append(A5S2NovelAUROC)
rowsCsv.append(A6S1NovelAUROC)
rowsCsv.append(A6S2NovelAUROC)


insertData.append(rowsCsv)

with open(outcsv, 'a', newline='', encoding='UTF8') as fileCsv:
  writer = csv.writer(fileCsv)

  writer.writerows(insertData)
print("Done")

# if __name__ == '__main__':
#     print("Done")