# Used to get image with specific label and counter in the testing dataset
# Used to showcase result of different variants of autoencoder 
# See results in "Results/Variant_n_autoencoder", where n is different variant number
def getImageLabel2(labelIn, counterIn, test_loader):

  counter = 0
  for image, label in test_loader:
    if(label == labelIn):
      if(counterIn == 1):
        break
      counterIn = counterIn - 1

  return image