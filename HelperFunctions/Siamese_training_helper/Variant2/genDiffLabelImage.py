import os
import sys
import random
sys.path.insert(0, os.getcwd())

from getImageForNoveltyCollection import getImageForNoveltyCollection


def genDiffLabelImage(label, noveltyTrainImage, numberOfClusters2):
  imageCollection = []
  randomImageIndex = random.randint(0, 49) 
  
  for indexClass in range(numberOfClusters2):
    if(indexClass == label.item()):
      q=1
    else:
      imageCollection.append(getImageForNoveltyCollection(indexClass, randomImageIndex, noveltyTrainImage))
  return imageCollection