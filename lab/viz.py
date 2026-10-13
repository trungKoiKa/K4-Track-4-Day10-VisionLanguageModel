"""Vẽ box đúng (xanh) và box dự đoán (đỏ) lên ảnh test, chọn cả ca đúng lẫn sai.

python -m lab.viz --pred work/e3.jsonl --name E3 --out report/figures --n-ok 3 --n-bad 5
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from lab.labels import box_to_pixels
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output

GT_COLOR, PRED_COLOR = (0, 200, 0), (230, 30, 30)


def classify(entry, raw, coord="norm1000"):
    """Trả về (trạng thái, pred_norm1000, (tp, fp, fn))."""
    parsed = parse_output(raw)
    pred = [] if parsed is None else _to_norm1000(parsed, coord, entry["width"], entry["height"])
    per, _ = _match(_gt_boxes(entry), pred)
    tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
    if parsed is None:
        status = "JSON hỏng"
    elif fp == 0 and fn == 0:
        status = "đúng hoàn toàn" if entry["objects"] else "đúng (không có đối tượng)"
    else:
        status = f"sai: TP={tp} FP={fp} FN={fn}"
    return status, pred, (tp, fp, fn)


def draw(entry, pred, status, name, data_dir):
    img = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for g in _gt_boxes(entry):
        d.rectangle(box_to_pixels(g["bbox"], "norm1000", w, h), outline=GT_COLOR, width=2)
        d.text((g["bbox"][0] / 1000 * w + 2, g["bbox"][1] / 1000 * h + 2), g["label"], fill=GT_COLOR)
    for p in pred:
        d.rectangle(box_to_pixels(p["bbox"], "norm1000", w, h), outline=PRED_COLOR, width=2)
        d.text((p["bbox"][0] / 1000 * w + 2, p["bbox"][3] / 1000 * h - 12), p["label"], fill=PRED_COLOR)
    canvas = Image.new("RGB", (w, h + 22), (255, 255, 255))
    canvas.paste(img, (0, 22))
    ImageDraw.Draw(canvas).text(
        (4, 4), f"{name} | {entry['image_id']} | {status} | xanh=GT đỏ=dự đoán", fill=(0, 0, 0))
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--n-ok", type=int, default=3)
    ap.add_argument("--n-bad", type=int, default=5)
    a = ap.parse_args(argv)

    entries = read_jsonl(Path(a.data_dir) / "splits" / f"{a.split}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(a.pred)}
    ok, bad = [], []
    for e in entries:
        if e["image_id"] not in preds:
            continue
        status, pred, counts = classify(e, preds[e["image_id"]], a.coord)
        item = (e, pred, status, counts)
        if status.startswith("đúng") and e["objects"]:
            ok.append(item)
        elif not status.startswith("đúng"):
            bad.append(item)
    bad.sort(key=lambda it: -(it[3][1] + it[3][2]))   # lỗi nặng trước
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for tag, items, n in (("ok", ok, a.n_ok), ("bad", bad, a.n_bad)):
        step = max(1, len(items) // max(n, 1))
        for k, (e, pred, status, _) in enumerate(items[::step][:n]):
            path = out / f"{a.name.lower()}_{tag}{k + 1}_{e['image_id']}.png"
            draw(e, pred, status, a.name, a.data_dir).save(path)
            print("đã lưu", path)


if __name__ == "__main__":
    main()
