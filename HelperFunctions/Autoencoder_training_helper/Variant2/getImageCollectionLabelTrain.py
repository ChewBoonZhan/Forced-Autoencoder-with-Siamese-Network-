import torch
# Used in variant 2 autoencoder
# Used to get predetermined label image based on label
# Predetermined label image is used to train the decoder in the following ratio:
# 0.9 * Predetermined label image + 0.1 * image
def getImageCollectionLabelTrain(labelInCollection, train_other_loaderClass, imageCollectionForIndex, device):
  returnLabelCollection = []
  for label in labelInCollection:
    # print(label)
    label = train_other_loaderClass.index(label)
    returnLabelCollection.append(imageCollectionForIndex[label])

  returnLabelCollection= torch.tensor(returnLabelCollection).to(device)
  return returnLabelCollection