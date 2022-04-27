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

# lengthNovelClassCollection = 3
# novelClassCollection = []
# for index in range(lengthNovelClassCollection):
#   randomInt = random.randint(0, lengthClassToUse-1)
#   while randomInt in novelClassCollection:
#     randomInt = random.randint(0, lengthClassToUse-1)
#   novelClassCollection.append(randomInt)
novelClassCollection = [4, 5, 6]
lengthNovelClassCollection = len(novelClassCollection)

# lengthActualNovelClass = 3
# actualNovelClass = []
# for index in range(lengthActualNovelClass):
#   randomInt = random.randint(0, lengthClassToUse-1)
#   while (randomInt in novelClassCollection) or (randomInt in actualNovelClass):
#     randomInt = random.randint(0, lengthClassToUse-1)
#   actualNovelClass.append(randomInt)
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
# Testing dataset, contains all labels
print("\nProcessing Testing dataset, contains all labels, ie actual, novel and non-novel labels")
test_loader, test_label = processImage(test_X, 
                                       test_Y, 
                                       removeNovel = False,
                                       removeActualNovel = False, 
                                       novelClassCollection = novelClassCollection,
                                       actualNovelClass = actualNovelClass,
                                       classToUse = classToUse
                                       )

# Training dataset for autoencoder, contains non-novel label only
print("\nProcessing Train dataset, contains non-novel labels")
trainData, trainLabel = processImage(train_X, 
                                     train_Y, 
                                     removeNovel = True,
                                     removeActualNovel = True,
                                     novelClassCollection = novelClassCollection,
                                     actualNovelClass = actualNovelClass,
                                     classToUse = classToUse
                                    )

# Training dataset for Siamese Network, contains novel label and non-novel label. Does not have actual novel label
print("\nProcessing Train dataset, contains novel and non-novel labels")
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
# Test Data
test_loader = torch.tensor(test_loader)
test_label = torch.tensor(test_label)

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

# Test Data
test_loader = TensorDataset(test_loader, test_label)

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

# Test data with actual novel label, novel label and non-novel label
test_loader = DeviceDataLoader(test_loader, device)

# Test data with novel label and non-novel label
test_TrainOtherData = DeviceDataLoader(trainOtherData, device) 

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
knnModel3 = KMeans(n_clusters=numberOfClusters)

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
# Get average latent space of autoencoder for training
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


        ############ train supervised autoencoder
        # variant 2
        # get the random latent representation for the associated label
        encoderRepresentation = getEncoderLatentCollection_supervised(actualLabel, familiarClass, encoderLatentCollection, device)
        imagesCompare = getImageCollectionLabelTrain(actualLabel, train_other_loaderClass, imageCollectionForIndex, device).float()

        outEncoder = encoderNet2_supervised(imgs)

        lossEncoder_variant2 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp2_supervised.zero_grad()
        lossEncoder_variant2.backward()
        encoderOp2_supervised.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet2_supervised(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet2_supervised(outEncoder)
        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant2 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp2_supervised.zero_grad()
        lossDecoder_variant2.backward()
        decoderOp2_supervised.step()


        # Train the autoencoder variant 3 - use latent space from autoencoder averaged
        encoderRepresentation = getAutoencoderEncoderRepresentation_supervised(actualLabel, familiarClass, encoderAutoencoderRepresentation, device)

        outEncoder = encoderNet_supervised(imgs)
        lossEncoder_variant3 = nn.functional.mse_loss(outEncoder, encoderRepresentation)

        encoderOp_supervised.zero_grad()
        lossEncoder_variant3.backward()
        encoderOp_supervised.step()


        outEncoder = encoderNet_supervised(imgs)
        outEncoder = outEncoder.detach()
        outDecoder = decoderNet_supervised(outEncoder)

        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant3 = nn.functional.mse_loss(outDecoder, imagesCompare)

        decoderOp_supervised.zero_grad()
        lossDecoder_variant3.backward()
        decoderOp_supervised.step()



    
    print('\nEpoch {}: Variant 1 Loss {}'.format(epoch, lossAutoencoder))   
    print('Epoch {}: Variant 2 Encoder Loss {}'.format(epoch, lossEncoder_variant2))
    print('Epoch {}: Variant 2 Decoder Loss {}'.format(epoch, lossDecoder_variant2))

    print('Epoch {}: Variant 3 Encoder Loss {}'.format(epoch, lossEncoder_variant3))
    print('Epoch {}: Variant 3 Decoder Loss {}'.format(epoch, lossDecoder_variant3))

    print('Epoch {}: Variant 4 Encoder Loss {}'.format(epoch, lossEncoder_variant4))
    print('Epoch {}: Variant 4 Decoder Loss {}'.format(epoch, lossDecoder_variant4))

    print('Epoch {}: Variant 5 Encoder Loss {}'.format(epoch, lossEncoder_variant5))
    print('Epoch {}: Variant 5 Decoder Loss {}'.format(epoch, lossDecoder_variant5))
  
  # saving model
  torch.save(encoderNet.state_dict(), os.getcwd() + "/Data/normalEncoder.pth")
  torch.save(decoderNet.state_dict(), os.getcwd() + "/Data/normalDecoder.pth")

  torch.save(encoderNet2.state_dict(), os.getcwd() + "/Data/variant4Encoder.pth")
  torch.save(decoderNet2.state_dict(), os.getcwd() + "/Data/variant4Decoder.pth")

  torch.save(encoderNet3.state_dict(), os.getcwd() + "/Data/variant5Encoder.pth")
  torch.save(decoderNet3.state_dict(), os.getcwd() + "/Data/variant5Decoder.pth")

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
# Define Siamese Network and Contrastive Loss for training
novelKNN = siameseNetwork(number_of_conv_final_channel, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
novelOptimizerKNN = torch.optim.Adam(novelKNN.parameters(), lr=learning_rate)

novelSupervised= siameseNetwork(number_of_conv_final_channel, number_of_channel, conv_output_flatten, kernel_size).float().to(device)
novelOptimizerSupervised = torch.optim.Adam(novelSupervised.parameters(), lr=learning_rate)

lossFunction1 = ContrastiveLoss()


#####################################################################
# Training/Loading Siamese Network using Contrastive Loss
# if(os.path.exists(os.getcwd() + "/Data/siameseVariant2.pth") and 
#     os.path.exists(os.getcwd() + "/Data/siameseVariant1.pth")):
#   print("\nLoading Siamese Network\n")
#   novelKNN.load_state_dict(torch.load(os.getcwd() + "/Data/siameseVariant2.pth", map_location=device))
#   novelSupervised.load_state_dict(torch.load(os.getcwd() + "/Data/siameseVariant1.pth", map_location=device))
# else:
#   print("\nTraining Siamese Network\n")
    
#   num_epochs = 30

#   novel_loss = []
#   novel_loss2 = []


#   for epoch in tqdm(range(0, num_epochs), desc ="Training Siamese Network"):
    
#     for idx, data in enumerate(train_other_loader, 0):

#         novelKNN.train()

#         novelSupervised.train()

#         imgs, actualLabel = data
#         imgs = imgs.float()

#         ## Train KNN siamese network

#         imgsTrain1, imgsTrain2, labelTrain = genImageLabelDataset(imgs, knnModel2, noveltyTrainImage, numberOfClusters2, device)

#         labelTrain = labelTrain.unsqueeze(0).permute(1,0).float().to(device)
        
#         out1, out2 = novelKNN(imgsTrain1,imgsTrain2)

#         lossKNN = lossFunction1(out1, out2, labelTrain)

#         novelOptimizerKNN.zero_grad()
#         lossKNN.backward()
#         novelOptimizerKNN.step()

#         ## Train supervised siamese network

#         imgsTrain1, imgsTrain2, labelTrain = genImageLabelDataset_supervised(imgs, actualLabel, device, imageCollectionForAllIndex, train_other_loaderClass)

#         labelTrain = labelTrain.unsqueeze(0).permute(1,0).float().to(device)
        
#         out1, out2 = novelSupervised(imgsTrain1,imgsTrain2)

#         lossSupervised = lossFunction1(out1, out2, labelTrain)

#         novelOptimizerSupervised.zero_grad()
#         lossSupervised.backward()
#         novelOptimizerSupervised.step()


#     print('Epoch {}: Loss for Siamese KNN {}'.format(epoch, lossKNN))
#     print('Epoch {}: Loss for SIamese sup {}'.format(epoch, lossSupervised))

#####################################################################
# Check accuracy of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# Variant 1 Autoencoder
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

  variant1Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 1 Accuracy: %f"%variant1Accuracy)


# Variant 2
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



  variant2Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 2 Accuracy: %f"%variant2Accuracy)

# Variant 3
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

  variant3Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 3 Accuracy: %f"%variant3Accuracy)

# Variant 4
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



  variant4Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 4 Accuracy: %f"%variant4Accuracy)

# Variant 5

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



  variant5Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 5 Accuracy: %f"%variant5Accuracy)

# Variant 6

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



  variant6Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 6 Accuracy: %f"%variant6Accuracy)

# Variant 7

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



  variant7Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 7 Accuracy: %f"%variant7Accuracy)


# Variant 8

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



  variant8Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 8 Accuracy: %f"%variant8Accuracy)


# Variant 9

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



  variant9Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 9 Accuracy: %f"%variant9Accuracy)


# Variant 10

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



  variant10Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 10 Accuracy: %f"%variant10Accuracy)

#####################################################################
# Check AUROC of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# variant 1

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

variant1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant1FPR, variant1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)
print("Method 1 Variant 1 AUROC: %f"%variant1AUROC)

# variant 2
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

variant2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant2FPR, variant2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 2 AUROC: %f"%variant2AUROC)

# variant 3

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

variant3AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant3FPR, variant3TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 3 AUROC: %f"%variant3AUROC)

# variant 4
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

variant4AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant4FPR, variant4TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 4 AUROC: %f"%variant4AUROC)

# variant 5
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

variant5AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant5FPR, variant5TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 5 AUROC: %f"%variant5AUROC)

# variant 6
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

variant6AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant6FPR, variant6TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 6 AUROC: %f"%variant6AUROC)

# variant 7
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

variant7AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant7FPR, variant7TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 7 AUROC: %f"%variant7AUROC)

# variant 8
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

variant8AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant8FPR, variant8TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 1 Variant 8 AUROC: %f"%variant8AUROC)

# variant 9
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

    
variant9AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant9FPR, variant9TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 Variant 9 AUROC: %f"%variant9AUROC)

# variant 10
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

    
variant10AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant10FPR, variant10TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 1 Variant 10 AUROC: %f"%variant10AUROC)

#####################################################################
# Generate AUROC graph
print("\nGenerating AUROC graph for Method 1\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Method_1_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Method_1_AUROC/")

plt.title("ROC Curve Actual Novel Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(variant1FPR, variant1TPR, label = "Variant 1")
plt.plot(variant2FPR, variant2TPR, label = "Variant 2")
plt.plot(variant3FPR, variant3TPR, label = "Variant 3")
plt.plot(variant4FPR, variant4TPR, label = "Variant 4")
plt.plot(variant5FPR, variant5TPR, label = "Variant 5")
plt.plot(variant6FPR, variant6TPR, label = "Variant 6")
plt.plot(variant7FPR, variant7TPR, label = "Variant 7")
plt.plot(variant8FPR, variant8TPR, label = "Variant 8")
plt.plot(variant9FPR, variant9TPR, label = "Variant 9")
plt.plot(variant10FPR, variant10TPR, label = "Variant 10")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train1/Method_1_AUROC/first_autoencoder_AUROC2.png")

#####################################################################
# Check accuracy of variants (Method 2)
# Method 2 consist of novel class and non-novel class only. It does not contain actual novel class
# The label used for testing is similar with the label used to train Siamese Network
# The data are different, with testing and training dataset

# Variant 1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant1NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 1 Accuracy: %f"%variant1NovelAccuracy)

# Variant 2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant2NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 2 Accuracy: %f"%variant2NovelAccuracy)

# Variant 3

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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

  variant3NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 3 Accuracy: %f"%variant3NovelAccuracy)

# Variant 4

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant4NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 4 Accuracy: %f"%variant4NovelAccuracy)

# Variant 5

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant5NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 5 Accuracy: %f"%variant5NovelAccuracy)

# Variant 6

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant6NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 6 Accuracy: %f"%variant6NovelAccuracy)

# Variant 7

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant7NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 7 Accuracy: %f"%variant7NovelAccuracy)

# Variant 8

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant8NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 8 Accuracy: %f"%variant8NovelAccuracy)

# Variant 9

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant9NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 9 Accuracy: %f"%variant9NovelAccuracy)

# Variant 10

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant10NovelAccuracy = correctElements/totalElements * 100

  print("Method 2 Variant 10 Accuracy: %f"%variant10NovelAccuracy)


#####################################################################
# Get AUROC for variants


# variant 1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant1NovelFPR, variant1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 1 AUROC: %f"%variant1NovelAUROC)


# variant 2

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant2NovelFPR, variant2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 2 AUROC: %f"%variant2NovelAUROC)

# variant 3
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant3NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant3NovelFPR, variant3NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 3 AUROC: %f"%variant3NovelAUROC)


# variant 4
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant4NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant4NovelFPR, variant4NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 4 AUROC: %f"%variant4NovelAUROC)

# variant 5
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant5NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant5NovelFPR, variant5NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 5 AUROC: %f"%variant5NovelAUROC)


# variant 6
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant6NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant6NovelFPR, variant6NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 6 AUROC: %f"%variant6NovelAUROC)

# variant 7
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant7NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant7NovelFPR, variant7NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 7 AUROC: %f"%variant7NovelAUROC)

# variant 8
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant8NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant8NovelFPR, variant8NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Method 2 Variant 8 AUROC: %f"%variant8NovelAUROC)

# variant 9
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_TrainOtherData, 0):
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

    
variant9NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant9NovelFPR, variant9NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 Variant 9 AUROC: %f"%variant9NovelAUROC)

# variant 10
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_TrainOtherData, 0):
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

    
variant10NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant10NovelFPR, variant10NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Method 2 Variant 10 AUROC: %f"%variant10NovelAUROC)

#####################################################################
# Generate AUROC Graph (Method 1)
print("\nGenerating AUROC graph for Method 2\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train1/Method_2_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train1/Method_2_AUROC/")

plt.title("ROC Curve Novel Class Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(variant1FPR, variant1TPR, label = "Variant 1")
plt.plot(variant2FPR, variant2TPR, label = "Variant 2")
plt.plot(variant3FPR, variant3TPR, label = "Variant 3")
plt.plot(variant4FPR, variant4TPR, label = "Variant 4")
plt.plot(variant5FPR, variant5TPR, label = "Variant 5")
plt.plot(variant6FPR, variant6TPR, label = "Variant 6")
plt.plot(variant7FPR, variant7TPR, label = "Variant 7")
plt.plot(variant8FPR, variant8TPR, label = "Variant 8")
plt.plot(variant9FPR, variant9TPR, label = "Variant 9")
plt.plot(variant10FPR, variant10TPR, label = "Variant 10")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train1/Method_2_AUROC/first_autoencoder_novel_AUROC.png")

#####################################################################
# Saving data to CSV
csv_header = [
              " ",
              " ",
              "Exp Variant 1 Accuracy", 
              "Exp Variant 2 Accuracy", 
              "Exp Variant 3 Accuracy", 
              "Exp Variant 4 Accuracy", 
              "Exp Variant 5 Accuracy", 
              "Exp Variant 6 Accuracy", 
              "Exp Variant 7 Accuracy", 
              "Exp Variant 8 Accuracy", 
              "Exp Variant 9 Accuracy", 
              "Exp Variant 10 Accuracy", 

              "Exp Variant 1 AUROC", 
              "Exp Variant 2 AUROC", 
              "Exp Variant 3 AUROC", 
              "Exp Variant 4 AUROC", 
              "Exp Variant 5 AUROC", 
              "Exp Variant 6 AUROC", 
              "Exp Variant 7 AUROC", 
              "Exp Variant 8 AUROC", 
              "Exp Variant 9 AUROC", 
              "Exp Variant 10 AUROC"
            ]
outcsv = os.getcwd() + "/Results/FYP_compare.csv"
if not os.path.exists(outcsv):
    with open(outcsv, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(csv_header)

insertData = []

rowsCsv = []

rowsCsv.append("Train 1")
rowsCsv.append("Method 1")
rowsCsv.append(variant1Accuracy)  
rowsCsv.append(variant2Accuracy)  
rowsCsv.append(variant3Accuracy)  
rowsCsv.append(variant4Accuracy)  
rowsCsv.append(variant5Accuracy)  
rowsCsv.append(variant6Accuracy)  
rowsCsv.append(variant7Accuracy)  
rowsCsv.append(variant8Accuracy)  
rowsCsv.append(variant9Accuracy)  
rowsCsv.append(variant10Accuracy)  

rowsCsv.append(variant1AUROC)
rowsCsv.append(variant2AUROC)
rowsCsv.append(variant3AUROC)
rowsCsv.append(variant4AUROC)
rowsCsv.append(variant5AUROC)
rowsCsv.append(variant6AUROC)
rowsCsv.append(variant7AUROC)
rowsCsv.append(variant8AUROC)
rowsCsv.append(variant9AUROC)
rowsCsv.append(variant10AUROC)

insertData.append(rowsCsv)
rowsCsv = []

rowsCsv.append("Train 1")
rowsCsv.append("Method 2")

rowsCsv.append(variant1NovelAccuracy)  
rowsCsv.append(variant2NovelAccuracy)  
rowsCsv.append(variant3NovelAccuracy)  
rowsCsv.append(variant4NovelAccuracy)  
rowsCsv.append(variant5NovelAccuracy)  
rowsCsv.append(variant6NovelAccuracy)  
rowsCsv.append(variant7NovelAccuracy)  
rowsCsv.append(variant8NovelAccuracy)  
rowsCsv.append(variant9NovelAccuracy)  
rowsCsv.append(variant10NovelAccuracy)  

rowsCsv.append(variant1NovelAUROC)
rowsCsv.append(variant2NovelAUROC)
rowsCsv.append(variant3NovelAUROC)
rowsCsv.append(variant4NovelAUROC)
rowsCsv.append(variant5NovelAUROC)
rowsCsv.append(variant6NovelAUROC)
rowsCsv.append(variant7NovelAUROC)
rowsCsv.append(variant8NovelAUROC)
rowsCsv.append(variant9NovelAUROC)
rowsCsv.append(variant10NovelAUROC)


insertData.append(rowsCsv)

with open(outcsv, 'a', newline='', encoding='UTF8') as fileCsv:
  writer = csv.writer(fileCsv)

  writer.writerows(insertData)




#####################################################################
# Redefining parameter for retraining 
print("\nRetraining autoencoder\n")
novelClassCollection = [4, 5, 6]
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
# Testing dataset, contains all labels
print("\nProcessing Testing dataset, contains all labels, ie actual, novel and non-novel labels")
test_loader, test_label = processImage(test_X, 
                                       test_Y, 
                                       removeNovel = False,
                                       removeActualNovel = False, 
                                       novelClassCollection = novelClassCollection,
                                       actualNovelClass = actualNovelClass,
                                       classToUse = classToUse
                                       )

# Training dataset for autoencoder, contains non-novel label only
print("\nProcessing Train dataset, contains non-novel labels")
trainData, trainLabel = processImage(train_X, 
                                     train_Y, 
                                     removeNovel = True,
                                     removeActualNovel = True,
                                     novelClassCollection = novelClassCollection,
                                     actualNovelClass = actualNovelClass,
                                     classToUse = classToUse
                                    )

# Training dataset for Siamese Network, contains novel label and non-novel label. Does not have actual novel label
print("\nProcessing Train dataset, contains novel and non-novel labels")
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
# Test Data
test_loader = torch.tensor(test_loader)
test_label = torch.tensor(test_label)

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

# Test Data
test_loader = TensorDataset(test_loader, test_label)

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

# Test data with actual novel label, novel label and non-novel label
test_loader = DeviceDataLoader(test_loader, device)

# Test data with novel label and non-novel label
test_TrainOtherData = DeviceDataLoader(trainOtherData, device) 

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
knnModel3 = KMeans(n_clusters=numberOfClusters)

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
# Get average latent space of autoencoder for training
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


        ############ train supervised autoencoder
        # variant 2
        # get the random latent representation for the associated label
        encoderRepresentation = getEncoderLatentCollection_supervised(actualLabel, familiarClass, encoderLatentCollection, device)
        imagesCompare = getImageCollectionLabelTrain(actualLabel, train_other_loaderClass, imageCollectionForIndex, device).float()

        outEncoder = encoderNet2_supervised(imgs)

        lossEncoder_variant2 = nn.functional.mse_loss(outEncoder, encoderRepresentation)
        
        encoderOp2_supervised.zero_grad()
        lossEncoder_variant2.backward()
        encoderOp2_supervised.step()

        ## retrain the decoder with fixed latent representation
        outEncoder = encoderNet2_supervised(imgs)
        outEncoder = outEncoder.detach()

        outDecoder = decoderNet2_supervised(outEncoder)
        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant2 = nn.functional.mse_loss(outDecoder, imagesCompare)
        
        decoderOp2_supervised.zero_grad()
        lossDecoder_variant2.backward()
        decoderOp2_supervised.step()


        # Train the autoencoder variant 3 - use latent space from autoencoder averaged
        encoderRepresentation = getAutoencoderEncoderRepresentation_supervised(actualLabel, familiarClass, encoderAutoencoderRepresentation, device)

        outEncoder = encoderNet_supervised(imgs)
        lossEncoder_variant3 = nn.functional.mse_loss(outEncoder, encoderRepresentation)

        encoderOp_supervised.zero_grad()
        lossEncoder_variant3.backward()
        encoderOp_supervised.step()


        outEncoder = encoderNet_supervised(imgs)
        outEncoder = outEncoder.detach()
        outDecoder = decoderNet_supervised(outEncoder)

        imagesCompare = ((0.9 * imagesCompare) + (0.1 * imgs)).clamp(0.0,1.0)
        lossDecoder_variant3 = nn.functional.mse_loss(outDecoder, imagesCompare)

        decoderOp_supervised.zero_grad()
        lossDecoder_variant3.backward()
        decoderOp_supervised.step()



    
    print('\nEpoch {}: Variant 1 Loss {}'.format(epoch, lossAutoencoder))   
    print('Epoch {}: Variant 2 Encoder Loss {}'.format(epoch, lossEncoder_variant2))
    print('Epoch {}: Variant 2 Decoder Loss {}'.format(epoch, lossDecoder_variant2))

    print('Epoch {}: Variant 3 Encoder Loss {}'.format(epoch, lossEncoder_variant3))
    print('Epoch {}: Variant 3 Decoder Loss {}'.format(epoch, lossDecoder_variant3))

    print('Epoch {}: Variant 4 Encoder Loss {}'.format(epoch, lossEncoder_variant4))
    print('Epoch {}: Variant 4 Decoder Loss {}'.format(epoch, lossDecoder_variant4))

    print('Epoch {}: Variant 5 Encoder Loss {}'.format(epoch, lossEncoder_variant5))
    print('Epoch {}: Variant 5 Decoder Loss {}'.format(epoch, lossDecoder_variant5))
  
  # saving model
  torch.save(encoderNet.state_dict(), os.getcwd() + "/Data/retrain_normalEncoder.pth")
  torch.save(decoderNet.state_dict(), os.getcwd() + "/Data/retrain_normalDecoder.pth")

  torch.save(encoderNet2.state_dict(), os.getcwd() + "/Data/retrain_variant4Encoder.pth")
  torch.save(decoderNet2.state_dict(), os.getcwd() + "/Data/retrain_variant4Decoder.pth")

  torch.save(encoderNet3.state_dict(), os.getcwd() + "/Data/retrain_variant5Encoder.pth")
  torch.save(decoderNet3.state_dict(), os.getcwd() + "/Data/retrain_variant5Decoder.pth")

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
# Check accuracy of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# Variant 1 Autoencoder
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

  variant1Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 1 Accuracy: %f"%variant1Accuracy)


# Variant 2
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



  variant2Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 2 Accuracy: %f"%variant2Accuracy)

# Variant 3
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

  variant3Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 3 Accuracy: %f"%variant3Accuracy)

# Variant 4
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



  variant4Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 4 Accuracy: %f"%variant4Accuracy)

# Variant 5

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



  variant5Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 5 Accuracy: %f"%variant5Accuracy)

# Variant 6

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



  variant6Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 6 Accuracy: %f"%variant6Accuracy)

# Variant 7

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



  variant7Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 7 Accuracy: %f"%variant7Accuracy)


# Variant 8

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



  variant8Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 8 Accuracy: %f"%variant8Accuracy)


# Variant 9

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



  variant9Accuracy = correctElements/totalElements * 100

  print("Method 1 Variant 9 Accuracy: %f"%variant9Accuracy)


# Variant 10

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



  variant10Accuracy = correctElements/totalElements * 100

  print("Retrain Method 1 Variant 10 Accuracy: %f"%variant10Accuracy)

#####################################################################
# Check AUROC of network (Method 1)
# Dataset to check accuracy consist of non-novel label, novel label and actual novel label

# variant 1

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

variant1AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant1FPR, variant1TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)
print("Retrain Method 1 Variant 1 AUROC: %f"%variant1AUROC)

# variant 2
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

variant2AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant2FPR, variant2TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 2 AUROC: %f"%variant2AUROC)

# variant 3

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

variant3AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant3FPR, variant3TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 3 AUROC: %f"%variant3AUROC)

# variant 4
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

variant4AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant4FPR, variant4TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 4 AUROC: %f"%variant4AUROC)

# variant 5
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

variant5AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant5FPR, variant5TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 5 AUROC: %f"%variant5AUROC)

# variant 6
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

variant6AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant6FPR, variant6TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 6 AUROC: %f"%variant6AUROC)

# variant 7
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

variant7AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant7FPR, variant7TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 7 AUROC: %f"%variant7AUROC)

# variant 8
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

variant8AUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant8FPR, variant8TPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 1 Variant 8 AUROC: %f"%variant8AUROC)

# variant 9
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

    
variant9AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant9FPR, variant9TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Retrain Method 1 Variant 9 AUROC: %f"%variant9AUROC)

# variant 10
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

    
variant10AUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant10FPR, variant10TPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Retrain Method 1 Variant 10 AUROC: %f"%variant10AUROC)

#####################################################################
# Generate AUROC graph
print("\nGenerating AUROC graph for Method 1\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Method_1_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Method_1_AUROC/")

plt.title("ROC Curve Actual Novel Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(variant1FPR, variant1TPR, label = "Variant 1")
plt.plot(variant2FPR, variant2TPR, label = "Variant 2")
plt.plot(variant3FPR, variant3TPR, label = "Variant 3")
plt.plot(variant4FPR, variant4TPR, label = "Variant 4")
plt.plot(variant5FPR, variant5TPR, label = "Variant 5")
plt.plot(variant6FPR, variant6TPR, label = "Variant 6")
plt.plot(variant7FPR, variant7TPR, label = "Variant 7")
plt.plot(variant8FPR, variant8TPR, label = "Variant 8")
plt.plot(variant9FPR, variant9TPR, label = "Variant 9")
plt.plot(variant10FPR, variant10TPR, label = "Variant 10")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train2/Method_1_AUROC/first_autoencoder_AUROC2.png")

#####################################################################
# Check accuracy of variants (Method 2)
# Method 2 consist of novel class and non-novel class only. It does not contain actual novel class
# The label used for testing is similar with the label used to train Siamese Network
# The data are different, with testing and training dataset

# Variant 1

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant1NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 1 Accuracy: %f"%variant1NovelAccuracy)

# Variant 2

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant2NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 2 Accuracy: %f"%variant2NovelAccuracy)

# Variant 3

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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

  variant3NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 3 Accuracy: %f"%variant3NovelAccuracy)

# Variant 4

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant4NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 4 Accuracy: %f"%variant4NovelAccuracy)

# Variant 5

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant5NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 5 Accuracy: %f"%variant5NovelAccuracy)

# Variant 6

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant6NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 6 Accuracy: %f"%variant6NovelAccuracy)

# Variant 7

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant7NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 7 Accuracy: %f"%variant7NovelAccuracy)

# Variant 8

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant8NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 8 Accuracy: %f"%variant8NovelAccuracy)

# Variant 9

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant9NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 9 Accuracy: %f"%variant9NovelAccuracy)

# Variant 10

with torch.no_grad():
  totalElements = 0
  correctElements = 0

  for idx, data in enumerate(test_TrainOtherData, 0):
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



  variant10NovelAccuracy = correctElements/totalElements * 100

  print("Retrain Method 2 Variant 10 Accuracy: %f"%variant10NovelAccuracy)


#####################################################################
# Get AUROC for variants


# variant 1
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant1NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant1NovelFPR, variant1NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 1 AUROC: %f"%variant1NovelAUROC)


# variant 2

noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant2NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant2NovelFPR, variant2NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 2 AUROC: %f"%variant2NovelAUROC)

# variant 3
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant3NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant3NovelFPR, variant3NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 3 AUROC: %f"%variant3NovelAUROC)


# variant 4
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant4NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant4NovelFPR, variant4NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 4 AUROC: %f"%variant4NovelAUROC)

# variant 5
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant5NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant5NovelFPR, variant5NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 5 AUROC: %f"%variant5NovelAUROC)


# variant 6
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant6NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant6NovelFPR, variant6NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 6 AUROC: %f"%variant6NovelAUROC)

# variant 7
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant7NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant7NovelFPR, variant7NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 7 AUROC: %f"%variant7NovelAUROC)

# variant 8
noveltyActualNovel = []
noveltyPredictedNovel = []


with torch.no_grad():
  
  for idx, data in enumerate(test_TrainOtherData, 0):
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

variant8NovelAUROC = roc_auc_score(noveltyActualNovel,noveltyPredictedNovel)
variant8NovelFPR, variant8NovelTPR, _ = roc_curve(noveltyActualNovel,noveltyPredictedNovel)

print("Retrain Method 2 Variant 8 AUROC: %f"%variant8NovelAUROC)

# variant 9
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_TrainOtherData, 0):
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

    
variant9NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant9NovelFPR, variant9NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Retrain Method 2 Variant 9 AUROC: %f"%variant9NovelAUROC)

# variant 10
# check accuracy of trained model~!
forcedNoveltyActualLabel = []
forcedNoveltyPredictedLabel = []

with torch.no_grad():
  totalElements = 0
  correctElements = 0
  for idx, data in enumerate(test_TrainOtherData, 0):
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

    
variant10NovelAUROC = roc_auc_score(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)
variant10NovelFPR, variant10NovelTPR, _ = roc_curve(forcedNoveltyActualLabel,forcedNoveltyPredictedLabel)

print("Retrain Method 2 Variant 10 AUROC: %f"%variant10NovelAUROC)

#####################################################################
# Generate AUROC Graph (Method 1)
print("\nGenerating AUROC graph for Method 2\n")
if(not os.path.isdir(os.getcwd() + "/Results/Train2/Method_2_AUROC/") ):
  os.mkdir(os.getcwd() + "/Results/Train2/Method_2_AUROC/")

plt.title("ROC Curve Novel Class Novelty Estimation")
plt.xlabel("False Positive Rates")
plt.ylabel("True Positive Rates")
plt.plot(variant1FPR, variant1TPR, label = "Variant 1")
plt.plot(variant2FPR, variant2TPR, label = "Variant 2")
plt.plot(variant3FPR, variant3TPR, label = "Variant 3")
plt.plot(variant4FPR, variant4TPR, label = "Variant 4")
plt.plot(variant5FPR, variant5TPR, label = "Variant 5")
plt.plot(variant6FPR, variant6TPR, label = "Variant 6")
plt.plot(variant7FPR, variant7TPR, label = "Variant 7")
plt.plot(variant8FPR, variant8TPR, label = "Variant 8")
plt.plot(variant9FPR, variant9TPR, label = "Variant 9")
plt.plot(variant10FPR, variant10TPR, label = "Variant 10")
plt.legend()
plt.savefig(os.getcwd() + "/Results/Train2/Method_2_AUROC/first_autoencoder_novel_AUROC.png")

#####################################################################
# Save data to CSV file

insertData = []

rowsCsv = []

rowsCsv.append("Train 2")
rowsCsv.append("Method 1")
rowsCsv.append(variant1Accuracy)  
rowsCsv.append(variant2Accuracy)  
rowsCsv.append(variant3Accuracy)  
rowsCsv.append(variant4Accuracy)  
rowsCsv.append(variant5Accuracy)  
rowsCsv.append(variant6Accuracy)  
rowsCsv.append(variant7Accuracy)  
rowsCsv.append(variant8Accuracy)  
rowsCsv.append(variant9Accuracy)  
rowsCsv.append(variant10Accuracy)  

rowsCsv.append(variant1AUROC)
rowsCsv.append(variant2AUROC)
rowsCsv.append(variant3AUROC)
rowsCsv.append(variant4AUROC)
rowsCsv.append(variant5AUROC)
rowsCsv.append(variant6AUROC)
rowsCsv.append(variant7AUROC)
rowsCsv.append(variant8AUROC)
rowsCsv.append(variant9AUROC)
rowsCsv.append(variant10AUROC)

insertData.append(rowsCsv)
rowsCsv = []

rowsCsv.append("Train 2")
rowsCsv.append("Method 2")

rowsCsv.append(variant1NovelAccuracy)  
rowsCsv.append(variant2NovelAccuracy)  
rowsCsv.append(variant3NovelAccuracy)  
rowsCsv.append(variant4NovelAccuracy)  
rowsCsv.append(variant5NovelAccuracy)  
rowsCsv.append(variant6NovelAccuracy)  
rowsCsv.append(variant7NovelAccuracy)  
rowsCsv.append(variant8NovelAccuracy)  
rowsCsv.append(variant9NovelAccuracy)  
rowsCsv.append(variant10NovelAccuracy)  

rowsCsv.append(variant1NovelAUROC)
rowsCsv.append(variant2NovelAUROC)
rowsCsv.append(variant3NovelAUROC)
rowsCsv.append(variant4NovelAUROC)
rowsCsv.append(variant5NovelAUROC)
rowsCsv.append(variant6NovelAUROC)
rowsCsv.append(variant7NovelAUROC)
rowsCsv.append(variant8NovelAUROC)
rowsCsv.append(variant9NovelAUROC)
rowsCsv.append(variant10NovelAUROC)


insertData.append(rowsCsv)

with open(outcsv, 'a', newline='', encoding='UTF8') as fileCsv:
  writer = csv.writer(fileCsv)

  writer.writerows(insertData)

print("Done")

# if __name__ == '__main__':
#     print("Done")