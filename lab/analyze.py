"""Số liệu phụ cho REPORT.md: split theo phiên, kích thước box theo lớp, kiểm chứng lệch thang toạ độ.

python -m lab.analyze --pred work/e3.jsonl --out report/analysis.json
"""
import argparse
import json
import statistics
from pathlib import Path

from scripts import score
from scripts.labcommon import CLASSES, read_jsonl


def split_stats(data_dir):
    out, all_areas = {}, {c: [] for c in CLASSES}
    for s in ("train", "val", "test"):
        rows = read_jsonl(Path(data_dir) / "splits" / f"{s}.jsonl")
        areas = {c: [] for c in CLASSES}
        for r in rows:
            for o in r["objects"]:
                _, _, w, h = o["yolo"]
                areas[o["label"]].append(w * h)
                all_areas[o["label"]].append(w * h)
        out[s] = {"images": len(rows), "sessions": len({r["session"] for r in rows}),
                  "boxes": {c: len(a) for c, a in areas.items()},
                  "median_area_pct": {c: _pct(statistics.median(a)) if a else None
                                      for c, a in areas.items()}}
    out["all"] = {"median_area_pct": {c: _pct(statistics.median(a)) for c, a in all_areas.items() if a}}
    return out


def _pct(v):
    return round(100 * v, 3)


def misread_scale(data_dir, pred_path, split="test"):
    """Chấm CÙNG một file dự đoán (thang 0–1000) theo hai cách hiểu toạ độ."""
    entries = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    res = {}
    for coord in ("norm1000", "pixel"):
        r = score.evaluate(entries, preds, coord=coord, n_boot=200)
        res[coord] = {"f1": r["micro"]["f1"], "precision": r["micro"]["precision"],
                      "recall": r["micro"]["recall"], "tp": r["micro"]["tp"]}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="report/analysis.json")
    a = ap.parse_args(argv)
    res = {"splits": split_stats(a.data_dir), "scale_misread": misread_scale(a.data_dir, a.pred)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
