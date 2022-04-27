import torch
# Get random label representation based on input label
# Input label is used to determine its index in "familiarClass"
# The index is then used to determine the average predetermined latent space based on trained normal autoencoder
# Used in Variant 3 Autoencoder
def getAutoencoderEncoderRepresentation_supervised(labelCollection, familiarClass, encoderAutoencoderRepresentation, device):   # CHANGED
  encoderLatentCollectionReturn = []
  
  for label in labelCollection:
    label = familiarClass.index(label)
    encoderLatentCollectionReturn.append(encoderAutoencoderRepresentation[label])
  
  encoderLatentCollectionReturn = torch.tensor(encoderLatentCollectionReturn).to(device).float()
  return encoderLatentCollectionReturn