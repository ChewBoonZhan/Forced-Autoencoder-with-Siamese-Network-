import os
import sys
import random
sys.path.insert(0, os.getcwd())

from getImageLabelTrain import getImageLabelTrain

def genSameImage_supervised(label, imageCollectionForAllIndex):  # Changed
  randomImageIndex = random.randint(0, 99)
  return getImageLabelTrain(label.item(), randomImageIndex, imageCollectionForAllIndex)