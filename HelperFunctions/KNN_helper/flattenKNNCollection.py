import numpy as np
import torch
# flattens input image for KNN classification
# Used by Autoencoder Variant 4 and Siamese Variant 2 (to get flattened image of input image.)
def flattenKNNCollection(imageCollection):
  imageCollection = imageCollection.to("cpu").numpy()
  flattenedImages = []
  for image in imageCollection:
    imageShape = image.shape
    flattenedShape = imageShape[1] * imageShape[2]
    image = np.resize(image, (flattenedShape))
    flattenedImages.append(image)

  flattenedImages = np.array(flattenedImages)
  return flattenedImages