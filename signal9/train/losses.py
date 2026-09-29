"""Multi-task loss functions with supervised contrastive metric learning (SIGNAL INFINITY §2.2)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def hier_loss(logits_or_l1: dict[str, torch.Tensor] | torch.Tensor,
              targets_or_l2: dict[str, torch.Tensor] | torch.Tensor,
              l3_logits: torch.Tensor | None = None,
              l1_target: torch.Tensor | None = None,
              l2_target: torch.Tensor | None = None,
              l3_target: torch.Tensor | None = None,
              w_level: dict[str, float] | None = None,
              sample_w: torch.Tensor | None = None,
              ls: float = 0.05) -> torch.Tensor:
    """Partial-label hierarchical cross-entropy loss with masked auxiliary regression or positional inputs."""
    if isinstance(logits_or_l1, dict):
        logits = logits_or_l1
        targets = targets_or_l2
        weights = w_level or {"l1": 0.3, "l2": 1.0, "l3": 1.0}
        device = logits.get("l2", list(logits.values())[0]).device
        total = torch.zeros((), device=device)

        for lvl, w in weights.items():
            if lvl not in logits or lvl not in targets:
                continue
            y = targets[lvl]
            mask = (y >= 0)
            if mask.any():
                ce = F.cross_entropy(logits[lvl][mask].float(), y[mask], label_smoothing=ls, reduction="none")
                sw = sample_w[mask] if sample_w is not None else torch.ones_like(ce)
                weighted_ce = (ce * sw).sum() / sw.sum().clamp_min(1e-6)
                total = total + w * weighted_ce

        for aux in ("snr", "log_baud", "bw_frac"):
            if aux in logits and aux in targets:
                y_aux = targets[aux]
                mask_aux = torch.isfinite(y_aux)
                if mask_aux.any():
                    pred = logits[aux][mask_aux].float().squeeze(-1)
                    targ = y_aux[mask_aux].float()
                    total = total + 0.1 * F.smooth_l1_loss(pred, targ)

        if "is_outlier" in targets and "l2" in logits:
            outlier_mask = targets["is_outlier"].bool()
            if outlier_mask.any():
                outlier_logits = logits["l2"][outlier_mask].float()
                uniform_target = torch.full_like(outlier_logits, 1.0 / outlier_logits.shape[-1])
                oe_loss = -(uniform_target * F.log_softmax(outlier_logits, dim=-1)).sum(dim=-1).mean()
                total = total + 0.5 * oe_loss

        return total
    else:
        # Positional signature: l1_logits, l2_logits, l3_logits, l1_target, l2_target, l3_target
        loss_l1 = F.cross_entropy(logits_or_l1, l1_target)
        loss_l2 = F.cross_entropy(targets_or_l2, l2_target)
        loss_l3 = F.cross_entropy(l3_logits, l3_target)
        return loss_l1 + loss_l2 + loss_l3



class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss, the L_out form of Khosla et al. (NeurIPS 2020).

        L = mean_i  -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p/t) / sum_{a != i} exp(z_i.z_a/t) )

    The previous implementation computed ``log(sum_p exp / sum_a exp)`` -- the
    log of a sum over positives, without the 1/|P(i)| average.  That is the
    L_in variant, which the SupCon paper shows trains worse: its gradient is
    dominated by the single easiest positive, so it does not pull the whole
    class together.  It also exponentiated similarities directly; here every
    term goes through log-softmax, which is numerically stable at any
    temperature.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels >= 0
        if valid.sum() < 2:
            return features.sum() * 0.0
        z = F.normalize(features[valid], dim=1)
        y = labels[valid].view(-1, 1)
        n = z.shape[0]
        sim = (z @ z.T) / self.temperature
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        # log p(a | i) over all a != i.
        log_prob = sim.masked_fill(self_mask, float("-inf"))
        log_prob = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
        pos = (y == y.T) & ~self_mask
        n_pos = pos.sum(dim=1)
        has_pos = n_pos > 0
        if not has_pos.any():
            return features.sum() * 0.0
        mean_log_prob_pos = (log_prob.masked_fill(~pos, 0.0)).sum(dim=1)[has_pos] / n_pos[has_pos]
        return -mean_log_prob_pos.mean()


def build_family_structures(device: torch.device | str = "cpu"):
    """Builds lookup tensors mapping class indices to families and family masks."""
    from .labels import MOD_VOCAB, FAMILY
    families = sorted(list(set(FAMILY.values())))
    fam_to_idx = {f: i for i, f in enumerate(families)}

    n_classes = len(MOD_VOCAB)
    n_families = len(families)

    class_to_family = torch.tensor(
        [fam_to_idx[FAMILY[m]] for m in MOD_VOCAB],
        dtype=torch.long, device=device
    )
    family_mask = torch.zeros((n_families, n_classes), dtype=torch.bool, device=device)
    for c_idx, m in enumerate(MOD_VOCAB):
        f_idx = fam_to_idx[FAMILY[m]]
        family_mask[f_idx, c_idx] = True

    return class_to_family, family_mask


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, ls: float = 0.0) -> torch.Tensor:
    """Cross-entropy over the rows whose target is a real class (>= 0)."""
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask], label_smoothing=ls)


def _masked_hierarchical_ce(logits: torch.Tensor,
                            target: torch.Tensor,
                            class_to_family: torch.Tensor | None = None,
                            family_mask: torch.Tensor | None = None,
                            lambda_intra: float = 1.0,
                            ls: float = 0.05) -> torch.Tensor:
    """Combines global flat cross-entropy with intra-family masked cross-entropy."""
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0.0

    v_logits = logits[mask]
    v_target = target[mask]
    loss_global = F.cross_entropy(v_logits, v_target, label_smoothing=ls)

    if lambda_intra <= 0.0 or class_to_family is None or family_mask is None:
        return loss_global

    if v_logits.shape[-1] != family_mask.shape[-1]:
        return loss_global

    device = v_logits.device
    c2f = class_to_family.to(device)
    fmask = family_mask.to(device)

    in_vocab = (v_target < len(c2f))
    if not in_vocab.any():
        return loss_global

    v_logits_sub = v_logits[in_vocab]
    v_target_sub = v_target[in_vocab]
    batch_fams = c2f[v_target_sub]
    batch_mask = fmask[batch_fams]

    masked_logits = v_logits_sub.masked_fill(~batch_mask, float("-inf"))
    # Note: label_smoothing MUST be 0.0 here because -inf masked positions produce inf with label_smoothing > 0
    loss_intra = F.cross_entropy(masked_logits, v_target_sub, label_smoothing=0.0)
    return loss_global + lambda_intra * loss_intra


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss over the rows whose target is finite (NaN = not labelled)."""
    target = target.to(pred.dtype)
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target[mask], delta=delta)


class SignalInfinityLoss(nn.Module):
    """Consolidated multi-task loss (§2.2), with every term masked to labelled rows."""
    def __init__(self, weights: dict[str, float] | None = None, lambda_intra: float = 1.0):
        super().__init__()
        self.supcon = SupervisedContrastiveLoss(temperature=0.1)
        self.lambda_intra = lambda_intra
        try:
            self.class_to_family, self.family_mask = build_family_structures()
        except Exception:
            self.class_to_family, self.family_mask = None, None
        self.w = {
            "mod": 1.0, "symrate": 0.3, "rolloff": 0.2, "cfo": 0.2, "snr": 0.02,
            "fec": 0.5, "scrambler": 0.3, "presence": 0.5, "supcon": 0.2, "outlier": 0.5,
        }
        if weights:
            self.w.update(weights)

    def forward(self,
                preds: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor],
                embeddings: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        l_mod = _masked_hierarchical_ce(
            preds["mod_logits"], targets["mod"].long(),
            self.class_to_family, self.family_mask,
            lambda_intra=self.lambda_intra, ls=0.05
        )


        sym = targets["symrate"].to(preds["log_symrate"].dtype)
        log_sym = torch.where(torch.isfinite(sym) & (sym > 0), torch.log(sym.clamp_min(1e-12)),
                              torch.full_like(sym, float("nan")))
        l_symrate = _masked_huber(preds["log_symrate"], log_sym)
        l_rolloff = _masked_huber(preds["rolloff"], targets["rolloff"])
        l_cfo = _masked_huber(preds["cfo_norm"], targets["cfo_norm"])
        # SNR is in dB, so Huber's quadratic zone (delta=1) is +/-1 dB; the small
        # weight keeps its tens-of-dB early errors from swamping the other terms.
        l_snr = _masked_huber(preds["snr_db"], targets["snr_db"], delta=3.0)
        l_fec = _masked_ce(preds["fec_logits"], targets["fec"].long())
        l_scrambler = _masked_ce(preds["scrambler_logits"], targets["scrambler"].long())

        pres = targets["presence"]
        pmask = pres >= 0
        l_presence = (F.binary_cross_entropy_with_logits(preds["presence_logit"][pmask], pres[pmask].float())
                      if pmask.any() else preds["presence_logit"].sum() * 0.0)

        l_supcon = preds["mod_logits"].sum() * 0.0
        if embeddings is not None:
            l_supcon = self.supcon(embeddings, targets["mod"].long())

        l_outlier = preds["mod_logits"].sum() * 0.0
        if "is_outlier" in targets:
            om = targets["is_outlier"].bool()
            if om.any():
                logp = F.log_softmax(preds["mod_logits"][om], dim=-1)
                l_outlier = -logp.mean()      # cross-entropy to the uniform distribution

        w = self.w
        total = (w["mod"] * l_mod + w["symrate"] * l_symrate + w["rolloff"] * l_rolloff
                 + w["cfo"] * l_cfo + w["snr"] * l_snr + w["fec"] * l_fec
                 + w["scrambler"] * l_scrambler + w["presence"] * l_presence
                 + w["supcon"] * l_supcon + w["outlier"] * l_outlier)

        return {
            "loss": total, "l_mod": l_mod, "l_symrate": l_symrate, "l_rolloff": l_rolloff,
            "l_cfo": l_cfo, "l_snr": l_snr, "l_fec": l_fec, "l_scrambler": l_scrambler,
            "l_presence": l_presence, "l_supcon": l_supcon, "l_outlier": l_outlier,
        }
