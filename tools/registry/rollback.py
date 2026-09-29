"""Single-command deterministic model rollback (SIGNAL INFINITY §4.4)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def rollback_model(registry_dir: str | Path, to_model_id: str | None = None, reason: str = "") -> bool:
    reg_p = Path(registry_dir)
    manifest_file = reg_p / "registry.json"

    if not manifest_file.exists():
        print(f"ERROR: Registry manifest {manifest_file} does not exist.")
        return False

    registry = json.loads(manifest_file.read_text(encoding="utf-8"))
    current = registry.get("active_model", None)
    target = to_model_id or registry.get("previous_model", None)

    if not target:
        print("ERROR: No valid target model found for rollback.")
        return False

    target_ckpt = reg_p / f"{target}.ckpt"
    if not target_ckpt.exists():
        print(f"ERROR: Target checkpoint {target_ckpt} missing.")
        return False

    registry["active_model"] = target
    registry["previous_model"] = current
    registry.setdefault("rollback_log", []).append({
        "from": current,
        "to": target,
        "reason": reason,
    })

    manifest_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"ROLLBACK SUCCESSFUL: Reverted from '{current}' to '{target}'. Reason: '{reason}'")
    return True


def main():
    parser = argparse.ArgumentParser(description="Roll back to previous model version")
    parser.add_argument("--registry", type=str, default="models/registry", help="Registry directory")
    parser.add_argument("--to", type=str, default=None, help="Target model ID")
    parser.add_argument("--reason", type=str, default="Diagnostic trigger", help="Rollback reason")
    args = parser.parse_args()

    rollback_model(args.registry, to_model_id=args.to, reason=args.reason)


if __name__ == "__main__":
    main()
