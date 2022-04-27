import os
import sys
import numpy as np
import torch
import random
sys.path.insert(0, os.getcwd())

from genSameImage_supervised import genSameImage_supervised
from genDiffLabelImage_supervised import genDiffLabelImage_supervised

def genImageLabelDataset_supervised(imageCollection, labelCollection, device, imageCollectionForAllIndex, train_other_loaderClass):  # Changed
  numberOfImages = imageCollection.shape[0]
  returnImage1Collection = []
  returnImage2Collection = []
  returnLabelCollection = []
  for index in range(numberOfImages):
    labelIndex = labelCollection[index]
    imageIndex = imageCollection[index]
    sameImage = genSameImage_supervised(labelIndex, imageCollectionForAllIndex)
    # sameImageCombine = torch.cat((imageIndex.to("cpu"), sameImage.to("cpu")), 0)
    # print(sameImageCombine.shape)
    returnImage1Collection.append(imageIndex.to("cpu").numpy().tolist())
    returnImage2Collection.append(sameImage.to("cpu").numpy().tolist())

    # append 0 for no difference in loss
    returnLabelCollection.append(0)

    diffLabelImage = genDiffLabelImage_supervised(labelIndex, imageCollectionForAllIndex, train_other_loaderClass)
    for imageDiffLabelOne in diffLabelImage:
      # diffImageCombine = torch.cat((imageIndex.to("cpu"), imageDiffLabelOne.to("cpu")), 0)
      # print(diffImageCombine.shape)

      returnImage1Collection.append(imageIndex.to("cpu").numpy().tolist())
      returnImage2Collection.append(imageDiffLabelOne.to("cpu").numpy().tolist())

      # append 1 for difference in loss
      returnLabelCollection.append(1)

  returnImage1Collection = torch.tensor(returnImage1Collection).float().to(device)
  returnImage2Collection = torch.tensor(returnImage2Collection).float().to(device)
  returnLabelCollection = torch.tensor(returnLabelCollection).float().to(device)


  return (returnImage1Collection, returnImage2Collection, returnLabelCollection)