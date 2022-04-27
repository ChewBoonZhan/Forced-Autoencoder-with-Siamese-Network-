import torch
import torch.nn.functional as F
class ContrastiveLoss(torch.nn.Module):
  def __init__(self, margin = 2.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
  
  def forward(self, output1, output2, label):
    # Calculate euclidian distance 
    euclidian_distance = F.pairwise_distance(output1, output2, keepdim = True)

    # Calculate contrastive loss
    loss_contrastive = torch.mean((1-label) * torch.pow(euclidian_distance, 2)+
                                  (label) * torch.pow(torch.clamp(self.margin - euclidian_distance, min = 0.0), 2))
    
    return loss_contrastive