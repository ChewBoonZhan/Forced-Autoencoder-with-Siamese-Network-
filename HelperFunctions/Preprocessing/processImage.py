import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

def processImage(imageCollection, labelCollection, removeNovel, removeActualNovel, novelClassCollection, actualNovelClass, classToUse):
  # states the number of images is required for familiar and non-familiar images
  
  numberOfImages = labelCollection.shape[0]
  outputImages = []
  outputLabels = []

  for index in tqdm(range(0, numberOfImages), "Processing Images"):
    
    indexNotInNovelCollection = (not (labelCollection[index] in novelClassCollection))
    indexNotInActualNovelCollection = (not (labelCollection[index] in actualNovelClass))
    
    if(not removeNovel):
      indexNotInNovelCollection = True

    if(not removeActualNovel):
      indexNotInActualNovelCollection = True

    if((labelCollection[index] in classToUse)):
      if(indexNotInActualNovelCollection):
        if indexNotInNovelCollection:
          
          image = imageCollection[index]

          image = image/255

          image = np.array(torch.tensor(image).unsqueeze(0))

          outputImages.append(image)
          outputLabels.append(labelCollection[index])
          ## end adding image and index into the list


  return (outputImages, outputLabels)
  