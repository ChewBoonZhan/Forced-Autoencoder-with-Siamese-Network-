import os
import sys
import random
sys.path.insert(0, os.getcwd())

from getImageForNoveltyCollection import getImageForNoveltyCollection

def genSameImage(label, noveltyTrainImage):
  randomImageIndex = random.randint(0, 49)
  return getImageForNoveltyCollection(label.item(), randomImageIndex, noveltyTrainImage)