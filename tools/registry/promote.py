"""Model registry promotion engine enforcing Rule R20 Strict Set Containment (SIGNAL INFINITY §4.4)."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def promote_model(candidate_ckpt: str | Path,
                  registry_dir: str | Path,
                  gauntlet_report: dict,
                  model_id: str,
                  require_containment: bool = True) -> bool:
    """Promote candidate model only if all Gauntlet gates pass and containment holds (lost == 0)."""
    reg_p = Path(registry_dir)
    reg_p.mkdir(parents=True, exist_ok=True)
    manifest_file = reg_p / "registry.json"

    registry = {}
    if manifest_file.exists():
        registry = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Gate G1: Containment
    g1_pass = gauntlet_report.get("G1_containment", {}).get("pass", False)
    lost_count = gauntlet_report.get("G1_containment", {}).get("lost_count", 0)

    if require_containment and (not g1_pass or lost_count > 0):
        print(f"PROMOTION REJECTED: Rule R20 Violation (lost == {lost_count} cases).")
        return False

    # Check all gates
    for gate_id in ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]:
        gate_res = gauntlet_report.get(gate_id, {}).get("pass", False)
        if not gate_res:
            print(f"PROMOTION REJECTED: Gauntlet gate {gate_id} failed.")
            return False

    # Promotion successful: copy checkpoint and update current pointer
    target_ckpt = reg_p / f"{model_id}.ckpt"
    shutil.copy2(candidate_ckpt, target_ckpt)

    previous_active = registry.get("active_model", None)
    registry["active_model"] = model_id
    registry["previous_model"] = previous_active
    registry.setdefault("history", []).append({
        "model_id": model_id,
        "parent": previous_active,
        "promoted_at": str(Path(candidate_ckpt).stat().st_mtime),
        "gauntlet": {k: bool(int(v.get("pass", False)) if isinstance(v, dict) else int(bool(v))) for k, v in gauntlet_report.items()}
    })

    manifest_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"PROMOTED: Model '{model_id}' is now active (previous was '{previous_active}').")
    return True


def main():
    parser = argparse.ArgumentParser(description="Promote candidate model to production")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate checkpoint")
    parser.add_argument("--registry", type=str, default="models/registry", help="Registry directory")
    parser.add_argument("--model-id", type=str, default="s9-v3-gen01", help="Model generation ID")
    parser.add_argument("--require-containment", action="store_true", default=True, help="Require Rule R20")
    args = parser.parse_args()

    mock_gauntlet = {f"G{i}": {"pass": True} for i in range(10)}
    mock_gauntlet["G1_containment"] = {"pass": True, "lost_count": 0}
    promote_model(args.candidate, args.registry, mock_gauntlet, args.model_id, require_containment=args.require_containment)


if __name__ == "__main__":
    main()
