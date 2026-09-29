import sys
from pathlib import Path
sys.path.append(str(Path('AMC_RESEARCH_V5_4-extracted/AMC_RESEARCH_V5').resolve()))

import torch
import numpy as np
from torch.utils.data import DataLoader
from signal9.train.trainer import SyntheticRFDataset
from signal9.train.backbone import build_backbone
from signal9.train.heads import SignalInfinityMultiTaskHeads
from signal9.train.labels import MOD_VOCAB

def run_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_path = r"C:\Users\adity\Downloads\s9_hybrid_xcit_L128_last.ckpt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    architecture = "hybrid_xcit"
    expert_dim = int(ck.get("expert_dim", 11))
    
    backbone = build_backbone(architecture,
                              expert_dim=expert_dim,
                              stem_dim=ck.get("stem_dim", 128),
                              emb_dim=ck.get("emb_dim", 512))
    
    heads = SignalInfinityMultiTaskHeads(emb_dim=ck.get("emb_dim", 512), 
                                         n_mods=len(ck.get("class_names") or MOD_VOCAB),
                                         n_fecs=ck.get("n_fecs", 5),
                                         n_scramblers=ck.get("n_scramblers", 5))
    
    selected_backbone = ck.get("ema_backbone") or ck.get("backbone")
    backbone.load_state_dict(selected_backbone, strict=True)
    heads.load_state_dict(ck.get("ema_heads") or ck["heads"])
    
    backbone.to(device)
    heads.to(device)
    backbone.eval()
    heads.eval()
    
    # Generate completely new (unknown) synthetic data on-the-fly using a unique seed
    print("Generating unseen synthetic data on-the-fly (L=128, seed=9999)...")
    ds = SyntheticRFDataset(n_samples=25000, length=128, seed=9999)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    snr_correct = {}
    snr_total = {}
    
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            mod_labels = batch["mod"].to(device)
            expert = batch["expert"][:, :expert_dim].to(device)
            
            z = backbone(iq, expert)
            preds = heads(z)
            mod_preds = preds["mod_logits"].argmax(dim=1)
            
            correct += (mod_preds == mod_labels).sum().item()
            total += mod_labels.size(0)
            
            for p, l, s in zip(mod_preds, mod_labels, batch["snr_db"]):
                if not torch.isnan(s):
                    s_val = int(s.item())
                    if s_val not in snr_total:
                        snr_total[s_val] = 0
                        snr_correct[s_val] = 0
                    snr_total[s_val] += 1
                    if p == l:
                        snr_correct[s_val] += 1
                    
    print("\n================ Evaluation Results ================")
    print(f"Overall Accuracy: {correct/total:.4f} ({correct}/{total})")
    print("\nAccuracy by SNR:")
    for snr in sorted(snr_total.keys()):
        acc = snr_correct[snr] / snr_total[snr]
        print(f"  SNR {snr:3d} dB: {acc:.4f}  ({snr_correct[snr]}/{snr_total[snr]})")

if __name__ == "__main__":
    run_eval()
