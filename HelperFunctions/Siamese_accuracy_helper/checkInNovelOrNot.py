
def checkInNovelOrNot(labelIn, actualNovelClass, novelClassCollection):
  if(labelIn in actualNovelClass or labelIn in novelClassCollection):
    return True
  else:
    return False