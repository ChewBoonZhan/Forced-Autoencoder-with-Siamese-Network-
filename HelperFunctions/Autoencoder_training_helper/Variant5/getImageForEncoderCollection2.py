import torch
# Get predetermined image based on label determined by KNN
# Used in Variant 5 autoencoder
def getImageForEncoderCollection2(labelCollection, autoencoderTrainImage2, device):
  returnImages = []
  
  for label in labelCollection:
    
    returnImages.append(autoencoderTrainImage2[label].squeeze(0))

  returnImages = torch.tensor(returnImages).to(device)
  
  return returnImages