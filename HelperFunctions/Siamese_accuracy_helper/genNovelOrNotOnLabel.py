import os
import sys

sys.path.insert(0, os.getcwd())

from checkInNovelOrNot import checkInNovelOrNot

def genNovelOrNotOnLabel(labels, actualNovelClass, novelClassCollection):
  newOutputLabel = []
  for label in labels:
    if checkInNovelOrNot(label, actualNovelClass, novelClassCollection):
      newOutputLabel.append(1)  # 1 is novel
    else:
      newOutputLabel.append(0)  # 0 is not novel
  return newOutputLabel