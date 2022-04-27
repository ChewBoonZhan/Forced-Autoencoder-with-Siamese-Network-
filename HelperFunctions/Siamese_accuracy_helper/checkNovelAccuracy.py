def checkNovelAccuracy(actualOutput, expectedOutput):
  # actualOutput = np.array(actualOutput.to("cpu")).tolist()
  # actualOutputIndex = actualOutput[0].index(max(actualOutput[0]))
  
  compareOutput = 0
  if(actualOutput>=0.5):
    compareOutput = 1


  # if(actualOutputIndex == expectedOutput[0]):
  if(compareOutput== expectedOutput[0]):
    return True
  else:
    return False