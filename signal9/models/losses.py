"""ArcFace Angular Margin Loss and Burst BCE Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """ArcFace angular margin loss (Additive Angular Margin)."""
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and weights
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-9, 1))
        
        # Add angular margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encode labels (ignore -1)
        valid_mask = (label >= 0)
        if not valid_mask.any():
            return emb.sum() * 0.0
        
        output = cosine * self.s
        if valid_mask.any():
            one_hot = torch.zeros(cosine.size(), device=emb.device)
            valid_labels = label[valid_mask].long()
            one_hot[valid_mask] = torch.zeros((valid_mask.sum(), self.out_features), device=emb.device).scatter_(1, valid_labels.view(-1, 1), 1)
            
            output_margin = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output_margin * self.s
        
        return F.cross_entropy(output, label.long(), ignore_index=-1)

class Signal9Loss(nn.Module):
    """End-to-end hierarchical loss with masked auxiliary supervision.

    Auxiliary targets are optional for backward compatibility.  Missing
    values must be NaN (regression) or -1 (outlier), and contribute no loss.
    """
    def __init__(self, n_l1: int, n_l2: int, n_l3: int, emb_dim: int = 256,
                 snr_weight: float = 0.02, baud_weight: float = 0.05,
                 outlier_weight: float = 0.10, arcface_weight: float = 0.0):
        super().__init__()
        self.l1_loss = ArcFaceLoss(emb_dim, n_l1)
        self.l2_loss = ArcFaceLoss(emb_dim, n_l2)
        self.l3_loss = ArcFaceLoss(emb_dim, n_l3)
        self.burst_loss = nn.BCEWithLogitsLoss()
        self.snr_weight = float(snr_weight)
        self.baud_weight = float(baud_weight)
        self.outlier_weight = float(outlier_weight)
        self.arcface_weight = float(arcface_weight)

    def forward(self, preds: dict[str, torch.Tensor], 
                t_l1: torch.Tensor, t_l2: torch.Tensor, t_l3: torch.Tensor, 
                t_burst: torch.Tensor, t_snr: torch.Tensor | None = None,
                t_log_baud: torch.Tensor | None = None,
                t_outlier: torch.Tensor | None = None) -> torch.Tensor:
        
        emb = preds["embedding"]

        def masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            valid = target >= 0
            return (F.cross_entropy(logits[valid], target[valid].long())
                    if valid.any() else logits.sum() * 0.0)

        # Train the logits that Signal9Net actually returns.  The previous
        # implementation optimized only three private ArcFace classifiers,
        # leaving HierarchicalHeads.head_l1/l2/l3 without gradients.
        loss_l1 = masked_ce(preds["l1"], t_l1)
        loss_l2 = masked_ce(preds["l2"], t_l2)
        loss_l3 = masked_ce(preds["l3"], t_l3)
        loss_arc = emb.sum() * 0.0
        if self.arcface_weight > 0.0:
            loss_arc = (self.l1_loss(emb, t_l1) + self.l2_loss(emb, t_l2)
                        + self.l3_loss(emb, t_l3))
        
        loss_burst = self.burst_loss(preds["burst"].squeeze(-1), t_burst.float())

        zero = emb.sum() * 0.0
        loss_snr = zero
        if t_snr is not None:
            mask = torch.isfinite(t_snr)
            if mask.any():
                loss_snr = F.smooth_l1_loss(preds["snr"][mask], t_snr[mask].to(preds["snr"].dtype))

        loss_baud = zero
        if t_log_baud is not None:
            mask = torch.isfinite(t_log_baud)
            if mask.any():
                loss_baud = F.smooth_l1_loss(
                    preds["log_baud"][mask], t_log_baud[mask].to(preds["log_baud"].dtype)
                )

        loss_outlier = zero
        if t_outlier is not None:
            mask = t_outlier >= 0
            if mask.any():
                loss_outlier = F.binary_cross_entropy_with_logits(
                    preds["outlier"][mask], t_outlier[mask].to(preds["outlier"].dtype)
                )

        return (loss_l1 + loss_l2 + loss_l3 + self.arcface_weight * loss_arc
                + 0.05 * loss_burst
                + self.snr_weight * loss_snr
                + self.baud_weight * loss_baud
                + self.outlier_weight * loss_outlier)
