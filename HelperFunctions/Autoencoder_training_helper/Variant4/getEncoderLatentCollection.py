import torch
# used by variant 2 autoencoder to get randomly determined latent space
def getEncoderLatentCollection(labelCollection, encoderLatentCollection, device):
  encoderLatentCollectionReturn = []
  
  for label in labelCollection:
    
    encoderLatentCollectionReturn.append(encoderLatentCollection[label])
  
  encoderLatentCollectionReturn = torch.tensor(encoderLatentCollectionReturn).to(device).float()
  return encoderLatentCollectionReturn