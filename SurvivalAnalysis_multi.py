import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 新增的武器: Focal Loss for Multi-Label Classification ---
class FocalLoss(nn.Module):
    """
    一个适用于多标签分类场景的Focal Loss实现，支持掩码。
    
    它解决了二元交叉熵（BCE）在正负样本极不平衡时遇到的问题。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 0.0, reduction: str = 'mean'):
        """
        Args:
            alpha (float): 用于平衡正负样本权重的因子，通常取0.25。
            gamma (float): 聚焦参数。gamma > 0 会降低分类良好样本的损失，
                           使得模型更专注于困难样本。
            reduction (str): 指定应用于输出的规约方式: 'none' | 'mean' | 'sum'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels, masks=None):
        """
        Args:
            logits (torch.Tensor): 模型的原始输出，形状 [B, N]。
            labels (torch.Tensor): 真实标签（多热编码），形状 [B, N]。
            masks (torch.Tensor, optional): 用于忽略某些损失计算的掩码，形状 [B, N]。
                                            值为1的位置保留，值为0的位置忽略。
        
        Returns:
            torch.Tensor: 计算出的Focal Loss。
        """
        # 使用 BCEWithLogitsLoss 以保证数值稳定性
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        # 计算概率 pt
        p = torch.sigmoid(logits)
        pt = torch.where(labels == 1, p, 1 - p)
        
        # 计算 alpha 权重
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        
        # 计算 focal loss modulating factor
        modulating_factor = (1.0 - pt).pow(self.gamma)
        
        # 计算最终的 focal loss
        focal_loss = alpha_t * modulating_factor * bce_loss
        
        # 应用掩码
        if masks is not None:
            focal_loss = focal_loss * masks.float()
        
        # 应用规约
        if self.reduction == 'mean':
            # 关键：只对被掩码计算的部分求平均
            if masks is not None:
                return focal_loss.sum() / masks.sum().clamp(min=1e-8)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss


# --- 您已有的代码 (保留，用于对比或传统模型) ---
def cox_loss(log_risk, times, events, epsilon=1e-7):
    """
    计算Cox比例风险损失。
    
    Args:
        log_risk (torch.Tensor): 模型输出的对数风险，形状 [B, 1]。
        times (torch.Tensor): 事件/删失时间，形状 [B, 1]。
        events (torch.Tensor): 事件状态 (1=事件, 0=删失)，形状 [B, 1]。
    
    Returns:
        torch.Tensor: 单个标量，代表该批次的Cox损失。
    """
    # 按照生存时间降序排列，这对于计算风险集至关重要
    times, sort_indices = torch.sort(times.view(-1), descending=True)
    log_risk = log_risk.view(-1)[sort_indices]
    events = events.view(-1)[sort_indices]

    # 计算每个时间点的风险累积和 (log-sum-exp trick for stability)
    log_risk_exp = torch.exp(log_risk)
    risk_set_sum = torch.log(torch.cumsum(log_risk_exp, dim=0) + epsilon)
    
    # 只选择那些真正发生了事件的样本来计算损失
    observed_log_risk = log_risk[events.bool()]
    observed_risk_set = risk_set_sum[events.bool()]
    
    # 对所有发生了事件的样本求平均损失
    loss = - (observed_log_risk - observed_risk_set).mean()
    
    # 如果该批次没有事件发生，返回0损失
    if torch.isnan(loss):
        loss = torch.tensor(0.0, device=log_risk.device, requires_grad=True)
        
    return loss

def c_index(log_risk, times, events):
    """
    计算C-index (一致性指数)。
    注意: 这是一个PyTorch的简单实现，用于验证，数据量大时会慢。
    """
    n_observed = 0
    n_concordant = 0
    
    # 遍历所有发生了事件的样本 i
    for i in range(len(times)):
        if events[i] == 1:
            # 比较样本i和所有生存时间比它长的样本 j
            for j in range(len(times)):
                if times[j] > times[i]:
                    n_observed += 1
                    # 如果风险预测正确（风险高的样本i，其风险值也更高），则为一致
                    if log_risk[i] > log_risk[j]:
                        n_concordant += 1
                        
    if n_observed == 0:
        # 如果没有可比较的对，则无法计算，通常返回0.5表示随机
        return 0.5
    
    return n_concordant / n_observed