"""Process boxes with decoders and write SigMF annotations."""
import argparse
from pathlib import Path
import json
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.io.sigmf_io import read_sigmf
from signal9.channelize.channelizer import channelize
from signal9.label.decoders.direwolf import verify_direwolf
from signal9.label.decoders.multimon import verify_multimon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="JSON report from Signal9Engine")
    parser.add_argument("--sigmf", required=True, help="Target SigMF file to read IQ and write annotations")
    args = parser.parse_args()

    report_path = Path(args.report)
    sigmf_path = Path(args.sigmf)
    
    with open(report_path, "r") as f:
        report = json.load(f)

    detections = report.get("signals", [])
    print(f"Loaded {len(detections)} detections.")
    
    if not detections or not sigmf_path.exists():
        print("No detections or SigMF file missing.")
        return

    iq_mmap, meta = read_sigmf(sigmf_path)
    fs = meta.sample_rate

    annotations = []
    
    for det in detections:
        t_lo, t_hi = det["t_lo"], det["t_hi"]
        f_center, bw = det["center_freq"], det["bandwidth"]
        
        try:
            # Channelize the detection crop
            ch_sig, ch_fs = channelize(iq_mmap, fs, f_center, bw, t_lo, t_hi)
            
            # Map predictions to a potential decoder
            pred_l3 = det.get("classification", {}).get("l3", {}).get("prediction", "unknown")
            
            verified_label = None
            if "aprs" in pred_l3.lower() or "ax25" in pred_l3.lower():
                res = verify_direwolf(ch_sig, ch_fs)
                if res.decoded:
                    verified_label = "aprs"
            elif "poc" in pred_l3.lower() or "flex" in pred_l3.lower():
                res = verify_multimon(ch_sig, ch_fs, mode="POCSAG1200")
                if res.decoded:
                    verified_label = "pocsag"
            
            tier = "GOLD" if verified_label else "SILVER"
            final_label = verified_label if verified_label else pred_l3
            
            annotations.append({
                "core:sample_start": int(t_lo * fs),
                "core:sample_count": int((t_hi - t_lo) * fs),
                "core:freq_lower_edge": f_center - bw/2,
                "core:freq_upper_edge": f_center + bw/2,
                "core:description": f"[{tier}] {final_label}"
            })
            
        except Exception as e:
            print(f"Failed to decode track at {f_center}: {e}")

    # Write annotations to sigmf
    meta.annotations.extend(annotations)
    meta.to_file(sigmf_path.with_suffix(".sigmf-meta"))
    print(f"Wrote {len(annotations)} annotations to {sigmf_path.name}")

if __name__ == "__main__":
    main()
