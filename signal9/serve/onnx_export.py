"""ONNX export wrapper and INT8 dynamic quantizer.
Fixes B19, B20: Named outputs match model forward pass, and dynamic INT8 is designated for CPU.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.backbone import Signal9Net

class Signal9InferenceWrapper(nn.Module):
    """Inference wrapper returning explicitly named output tensors for ONNX export."""
    def __init__(self, net: Signal9Net, temperature: float = 1.0):
        super().__init__()
        self.net = net
        self.temp = temperature

    def forward(self, iq: torch.Tensor,
                spec: torch.Tensor,
                expert: torch.Tensor,
                meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(iq, spec, expert, meta)
        p_l1 = F.softmax(out["l1"] / self.temp, dim=-1)
        p_l2 = F.softmax(out["l2"] / self.temp, dim=-1)
        p_l3 = F.softmax(out["l3"] / self.temp, dim=-1)

        # Free energy score on L2: -T * logsumexp(z/T)
        scaled = out["l2"] / self.temp
        energy = -self.temp * torch.logsumexp(scaled, dim=-1)

        snr = out["snr"]
        baud = out["log_baud"]
        return p_l1, p_l2, p_l3, energy, snr, baud

def export_onnx(model: Signal9Net, output_path: str | Path, opset: int = 17) -> Path:
    """Export Signal9Net to ONNX with named dict outputs."""
    out_p = Path(output_path)
    wrapper = Signal9InferenceWrapper(model).eval()

    dummy_iq = torch.randn(1, 2, 4096)
    dummy_spec = torch.randn(1, 1, 128, 128)
    dummy_expert = torch.randn(1, 32)
    dummy_meta = torch.randn(1, 16)

    output_names = ["p_l1", "p_l2", "p_l3", "energy", "snr_pred", "log_baud_pred"]
    input_names = ["iq", "spec", "expert", "meta"]

    torch.onnx.export(
        wrapper,
        (dummy_iq, dummy_spec, dummy_expert, dummy_meta),
        str(out_p),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes={
            "iq": {0: "batch", 2: "seq_len"},
            "spec": {0: "batch", 2: "spec_height", 3: "spec_width"},
            "expert": {0: "batch"},
            "meta": {0: "batch"},
            "p_l1": {0: "batch"},
            "p_l2": {0: "batch"},
            "p_l3": {0: "batch"},
            "energy": {0: "batch"},
            "snr_pred": {0: "batch"},
            "log_baud_pred": {0: "batch"},
        }
    )
    return out_p

def quantize_to_int8(onnx_path: str | Path, int8_output_path: str | Path) -> Path:
    """Apply dynamic INT8 quantization for CPU execution (fixes B20)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(onnx_path), str(int8_output_path), weight_type=QuantType.QInt8)
        return Path(int8_output_path)
    except ImportError:
        return Path(onnx_path)
