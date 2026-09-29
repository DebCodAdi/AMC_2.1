"""Fusion of CFAR and YOLO bounding boxes using WBF/NMS."""
from __future__ import annotations
import numpy as np

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """Compute Intersection over Union of two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

def merge_boxes(cfar_boxes: list[dict], yolo_boxes: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Merge CFAR (fallback) and YOLO detections using NMS-style logic.
    Assumes dict format: {'f_lo', 'f_hi', 't_lo', 't_hi', 'conf', 'source'}
    """
    all_boxes = []
    
    # Normalize formats
    for b in cfar_boxes:
        all_boxes.append({
            'box': [b['t_lo'], b['f_lo'], b['t_hi'], b['f_hi']],
            'conf': b.get('conf', 0.5), # CFAR has no explicit conf, assign 0.5
            'source': 'cfar',
            'orig': b
        })
        
    for b in yolo_boxes:
        # Assuming yolo_boxes have already been mapped from pixels to t/f coords
        all_boxes.append({
            'box': [b['t_lo'], b['f_lo'], b['t_hi'], b['f_hi']],
            'conf': b.get('conf', 0.8), # YOLO usually higher conf
            'source': 'yolo',
            'orig': b
        })
        
    # Sort by confidence descending
    all_boxes = sorted(all_boxes, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while all_boxes:
        current = all_boxes.pop(0)
        keep.append(current['orig'])
        
        # Remove highly overlapping boxes
        all_boxes = [b for b in all_boxes if compute_iou(current['box'], b['box']) < iou_thresh]
        
    return keep
