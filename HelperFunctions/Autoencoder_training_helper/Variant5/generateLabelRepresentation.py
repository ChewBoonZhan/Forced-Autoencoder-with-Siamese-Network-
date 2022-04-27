import numpy as np
# Generate label prediction based on input image
# Generate latent space for input image, which is put into KNN Model for prediction
# Returns label prediction back
# Used in Autoencoder Variant 5
def generateLabelRepresentation(imgs, encoderNet0, knnModel3):
  
  outputLabel = []
  for image in imgs:
    image = image.unsqueeze(0)
    outputEncoder = encoderNet0(image)

    predictedLabel = knnModel3.predict(outputEncoder.detach().to("cpu").numpy().astype(np.float64))

    outputLabel.append(predictedLabel[0])

  return outputLabel