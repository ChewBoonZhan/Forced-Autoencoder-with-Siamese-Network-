import math
# Used in variant 4 and variant 2 autoencoder
# used to calculate distance between latent space randomly generated to make sure it is more than a threshold
def check_distance_between_latent(latentRepresentation, encoderLatentCollection, latent_space_features):
  # threshold manually determined here
  threshold = 8
  thresholdNotMet = False
  numberOfRepresentation = len(encoderLatentCollection)
  for index2 in range(numberOfRepresentation):
    encoderRepresentation = encoderLatentCollection[index2]
    finalSum = 0

    for index3 in range(latent_space_features):  
      temp = (latentRepresentation[index3] - encoderRepresentation[index3]) 
      temp = temp * temp
      finalSum = finalSum + temp
    finalSum = math.sqrt(finalSum)
    if(finalSum < threshold):
      thresholdNotMet = True
      break
  if(thresholdNotMet):
    return False
  else:
    return True
     