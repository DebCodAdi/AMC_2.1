import sys
from pathlib import Path
sys.path.append(str(Path('AMC_RESEARCH_V5_4-extracted/AMC_RESEARCH_V5').resolve()))

import torch
from torch.utils.data import DataLoader
from signal9.train.trainer import KaggleRadioMLDataset, evaluate
from signal9.train.inference import MultiTaskClassifier
from signal9.train.backbone import build_backbone

def run_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_path = r"C:\Users\adity\Downloads\s9_hybrid_xcit_L128_last.ckpt"
    print(f"Loading checkpoint {ckpt_path}...")
    
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Architecture in ckpt: {ck.get('architecture')}")
    
    try:
        classifier = MultiTaskClassifier.load(ckpt_path, device=device)
        print(f"Model loaded successfully using .load().")
    except Exception as e:
        print(f"Error loading model via .load(): {e}")
        # Manually construct
        print("Falling back to manual construction...")
        
        from signal9.train.heads import SignalInfinityMultiTaskHeads
        from signal9.train.labels import MOD_VOCAB
        
        names = ck.get("class_names") or list(MOD_VOCAB)
        architecture = "hybrid_xcit"
        expert_dim = int(ck.get("expert_dim", 32))
        
        backbone = build_backbone(architecture,
                                  expert_dim=expert_dim,
                                  stem_dim=ck.get("stem_dim", 128),
                                  emb_dim=ck.get("emb_dim", 512))
        heads = SignalInfinityMultiTaskHeads(emb_dim=ck.get("emb_dim", 512), n_mods=len(names),
                                             n_fecs=ck.get("n_fecs", 5),
                                             n_scramblers=ck.get("n_scramblers", 5))
        
        selected_backbone = ck.get("ema_backbone") or ck.get("backbone")
        backbone.load_state_dict(selected_backbone, strict=True)
        heads.load_state_dict(ck.get("ema_heads") or ck["heads"])
        
        backbone.to(device)
        heads.to(device)
        
        print("Manual construction successful!")
        classifier = None
        classifier_backbone = backbone
        classifier_heads = heads

    if classifier is not None:
        classifier_backbone = classifier.backbone
        classifier_heads = classifier.heads

    pkl_path = r"C:\Users\adity\Downloads\AMC_1-satvik\AMC_1-satvik\RML2016.10a_dict_optimized.pkl"
    print(f"Loading dataset {pkl_path}...")
    try:
        ds = KaggleRadioMLDataset(pkl_path, max_samples=50000)
        print(f"Dataset loaded. Total samples: {len(ds)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    print("Running evaluation...")
    
    try:
        metrics = evaluate(classifier_backbone, classifier_heads, loader, device)
        print("================ Evaluation Results ================")
        print(f"Overall Accuracy: {metrics.get('val_acc', 0):.4f}")
        
        snr_dict = metrics.get('val_acc_by_snr', {})
        if snr_dict:
            print("Accuracy by SNR:")
            for snr in sorted([int(k) for k in snr_dict.keys()]):
                print(f"  SNR {snr:3d} dB: {snr_dict[snr]:.4f}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_eval()
