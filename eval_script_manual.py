import sys
from pathlib import Path
sys.path.append(str(Path('AMC_RESEARCH_V5_4-extracted/AMC_RESEARCH_V5').resolve()))

import torch
from torch.utils.data import DataLoader
from signal9.train.trainer import KaggleRadioMLDataset
from signal9.train.inference import MultiTaskClassifier
from signal9.train.backbone import build_backbone
from signal9.train.heads import SignalInfinityMultiTaskHeads
from signal9.train.labels import MOD_VOCAB

def run_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_path = r"C:\Users\adity\Downloads\s9_hybrid_xcit_L128_best.ckpt"
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
    
    pkl_path = r"C:\Users\adity\Downloads\AMC_1-satvik\AMC_1-satvik\RML2016.10a_dict_optimized.pkl"
    ds = KaggleRadioMLDataset(pkl_path, max_samples=50000)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
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
            
            for p, l in zip(mod_preds, mod_labels):
                l_val = int(l.item())
                if l_val not in class_total:
                    class_total[l_val] = 0
                    class_correct[l_val] = 0
                class_total[l_val] += 1
                if p == l:
                    class_correct[l_val] += 1
                    
    print("================ Class-by-Class Accuracy ================")
    # Sort classes by accuracy descending
    results = []
    for cls_idx in class_total.keys():
        acc = class_correct[cls_idx] / class_total[cls_idx]
        mod_name = MOD_VOCAB[cls_idx] if cls_idx < len(MOD_VOCAB) else str(cls_idx)
        results.append((acc, mod_name, class_correct[cls_idx], class_total[cls_idx]))
        
    results.sort(key=lambda x: x[0], reverse=True)
    
    for acc, mod_name, c, t in results:
        print(f"{mod_name.rjust(10)}: {acc:.4f}  ({c}/{t})")

if __name__ == "__main__":
    run_eval()
