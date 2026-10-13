"""Reward IoU cho GRPO (GUIDE bước 9). Python thuần, test được không cần GPU."""
from scripts.score import iou_xyxy, parse_output

IOU_THR = 0.5


def reward(completion, gt_boxes):
    """gt_boxes: list[{"label", "bbox"}] thang 0–1000. Trả về số trong [0, 1]."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)
