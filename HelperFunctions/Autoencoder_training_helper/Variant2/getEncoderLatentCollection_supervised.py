import torch
# Get predetermined random latent space for variant 2 autoencoder
def getEncoderLatentCollection_supervised(labelCollection, familiarClass, encoderLatentCollection, device):
  encoderLatentCollectionReturn = []
  
  for label in labelCollection:
    label = familiarClass.index(label)
    encoderLatentCollectionReturn.append(encoderLatentCollection[label])
  
  encoderLatentCollectionReturn = torch.tensor(encoderLatentCollectionReturn).to(device).float()
  return encoderLatentCollectionReturn