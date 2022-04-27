import os
import sys
import random
sys.path.insert(0, os.getcwd())

from getImageLabelTrain import getImageLabelTrain

def genDiffLabelImage_supervised(label, imageCollectionForAllIndex, train_other_loaderClass):  # Changed
  imageCollection = []
  randomImageIndex = random.randint(0, 99)
  for indexClass in train_other_loaderClass:
    if(indexClass == label.item()):
      q=1
    else:
      imageCollection.append(getImageLabelTrain(indexClass, randomImageIndex, imageCollectionForAllIndex))
  return imageCollection