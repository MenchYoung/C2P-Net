import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 0.0, reduction: str = 'mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels, masks=None):

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        p = torch.sigmoid(logits)
        pt = torch.where(labels == 1, p, 1 - p)
        
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        
        modulating_factor = (1.0 - pt).pow(self.gamma)
        
        focal_loss = alpha_t * modulating_factor * bce_loss
        
        if masks is not None:
            focal_loss = focal_loss * masks.float()
        
        if self.reduction == 'mean':
            if masks is not None:
                return focal_loss.sum() / masks.sum().clamp(min=1e-8)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: 
            return focal_loss


def cox_loss(log_risk, times, events, epsilon=1e-7):

    times, sort_indices = torch.sort(times.view(-1), descending=True)
    log_risk = log_risk.view(-1)[sort_indices]
    events = events.view(-1)[sort_indices]

    log_risk_exp = torch.exp(log_risk)
    risk_set_sum = torch.log(torch.cumsum(log_risk_exp, dim=0) + epsilon)
    observed_log_risk = log_risk[events.bool()]
    observed_risk_set = risk_set_sum[events.bool()]
    
    loss = - (observed_log_risk - observed_risk_set).mean()
    
    if torch.isnan(loss):
        loss = torch.tensor(0.0, device=log_risk.device, requires_grad=True)
        
    return loss

def c_index(log_risk, times, events):

    n_observed = 0
    n_concordant = 0
    

    for i in range(len(times)):
        if events[i] == 1:

            for j in range(len(times)):
                if times[j] > times[i]:
                    n_observed += 1

                    if log_risk[i] > log_risk[j]:
                        n_concordant += 1
                        
    if n_observed == 0:

        return 0.5
    

    return n_concordant / n_observed
