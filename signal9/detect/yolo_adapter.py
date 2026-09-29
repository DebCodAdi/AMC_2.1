"""YOLO11n detector integration for class-agnostic signal bounding boxes."""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

class YoloDetector:
    """Wrapper for the YOLO11n object detector."""
    def __init__(self, model_path: str | Path | None = None, conf_thres: float = 0.25):
        self.conf_thres = conf_thres
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(model_path))
                self.model.to(self.device)
            except ImportError:
                print("Warning: ultralytics package not found. YOLO detector disabled.")

    def detect(self, P_db: np.ndarray, f: np.ndarray, t: np.ndarray) -> list[dict]:
        """Detect bounding boxes in a spectrogram and map to physical coords."""
        if self.model is None:
            return []

        # Normalize P_db to 0-255 uint8
        p_min, p_max = np.min(P_db), np.max(P_db)
        if p_max > p_min:
            img = ((P_db - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(P_db, dtype=np.uint8)

        # YOLO expects RGB images.
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        results = self.model(img, conf=self.conf_thres, verbose=False)
        boxes_out = []
        
        H, W = P_db.shape
        f_min, f_max = f[0], f[-1]
        t_min, t_max = t[0], t[-1]
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                x1, y1, x2, y2 = b
                
                # Map pixels to time (x) and frequency (y)
                t_lo = t_min + (x1 / W) * (t_max - t_min)
                t_hi = t_min + (x2 / W) * (t_max - t_min)
                f_lo = f_min + (y1 / H) * (f_max - f_min)
                f_hi = f_min + (y2 / H) * (f_max - f_min)
                
                bw = f_hi - f_lo
                boxes_out.append({
                    'f_lo': float(f_lo),
                    'f_hi': float(f_hi),
                    't_lo': float(t_lo),
                    't_hi': float(t_hi),
                    'center_freq': float(f_lo + bw/2),
                    'bandwidth': float(bw),
                    'duration': float(t_hi - t_lo),
                    'conf': float(c)
                })
        return boxes_out
