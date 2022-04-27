import torch
# Used in variant 4 autoencoder
# Get predetermined images based on label determined by KNN
def getImageForEncoderCollection(labelCollection, autoencoderTrainImage, device):
  returnImages = []
  for label in labelCollection:
    
    returnImages.append(autoencoderTrainImage[label])

  returnImages = torch.tensor(returnImages).to(device)
  return returnImages