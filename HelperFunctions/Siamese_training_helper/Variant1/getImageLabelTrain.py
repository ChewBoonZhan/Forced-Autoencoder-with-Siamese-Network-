# Returns image based on input label
# Image correspond to label in the dataset.
# Used to train variant 1 Supervised Siamese Network
def getImageLabelTrain(labelIn, counterIn, imageCollectionForAllIndex):

  return imageCollectionForAllIndex[labelIn][counterIn]
