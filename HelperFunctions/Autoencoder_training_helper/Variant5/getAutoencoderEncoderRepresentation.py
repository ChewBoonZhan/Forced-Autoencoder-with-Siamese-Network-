import torch
# Used in variant autoencoder 5
# Given input label, get the average fixed latent space determined based on pretrained normal autoencoder
# label is generated based on "generateLabelRepresentation"
def getAutoencoderEncoderRepresentation(labelCollection, autoencoderEncoderCollectionAvg, device):
  encoderLatentCollectionReturn = []
  
  for label in labelCollection:
    # label = familiarClass.index(label)
    encoderLatentCollectionReturn.append(autoencoderEncoderCollectionAvg[label])
  
  encoderLatentCollectionReturn = torch.tensor(encoderLatentCollectionReturn).to(device).float()
  return encoderLatentCollectionReturn