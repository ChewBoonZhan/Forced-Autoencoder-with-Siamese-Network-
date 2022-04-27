import torch
# Used to get image with label based on input label
# Images are classified into labels based on KNN
def getImageForNoveltyCollection(label, randomIndex, noveltyTrainImage):
  return torch.tensor(noveltyTrainImage[label][randomIndex])
