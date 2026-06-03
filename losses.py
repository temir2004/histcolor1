import torch
import torch.nn as nn

class HistogramLoss(nn.Module):
    def __init__(self, bins=32):
        super().__init__()
        self.bins = bins

    def forward(self, pred_ab, target_ab):
        # pred_ab, target_ab: (B,2,H,W)
        def get_hist(x):
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)
            hist = torch.zeros(B, C, self.bins, device=x.device)
            for b in range(B):
                for c in range(C):
                    values = x_flat[b, c]
                    minv, maxv = values.min(), values.max()
                    if maxv - minv > 1e-5:
                        indices = ((values - minv) / (maxv - minv) * (self.bins-1)).long()
                        hist[b, c] = torch.bincount(indices, minlength=self.bins).float()
                    else:
                        hist[b, c, 0] = len(values)
            return hist / (H*W)
        hist_pred = get_hist(pred_ab)
        hist_target = get_hist(target_ab)
        return torch.mean((hist_pred - hist_target) ** 2)

def combined_loss(pred_ab, target_ab, hist_fn=None):
    l1 = nn.functional.l1_loss(pred_ab, target_ab)
    hist = hist_fn(pred_ab, target_ab) if hist_fn is not None else 0
    return l1 + 0.05 * hist