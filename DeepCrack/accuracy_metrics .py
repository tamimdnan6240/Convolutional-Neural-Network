class accuracy:
  def dice_coefficient(pred, mask):
    smooth = 1e-5 #this is constant value to avoid divided by 0 error
    ## Flatten tensor
    pred = pred.view(-1)
    mask = mask.view(-1)
    # calculate intersection and union
    intersection = (pred.mask).sum()
    union = pred.sum() + mask.sum()
   # calculate dice cofficient 
    accuracy = (2 * intersection + smooth) / (union + smooth) ## 2 is coefficient, included in numerator to ensure dice coefficent is between o and 1
   ## accuracy means dice here  
    return 
   
   