"""Suy luận trên split -> jsonl {"image_id","raw"} (GUIDE bước 3, 7).

E0:    python -m lab.infer --out work/e0.jsonl
E1–E3: python -m lab.infer --run-dir <thư mục huấn luyện> --out work/e1.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from lab.labels import get_prompt
from lab.model import load_extra_trainable, load_model, load_processor
from scripts.labcommon import read_jsonl, write_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def load_for_inference(run_dir=None, model_id=MODEL_ID):
    processor = load_processor(model_id)
    model = load_model(model_id)
    if run_dir:
        from peft import PeftModel
        run_dir = Path(run_dir)
        model = PeftModel.from_pretrained(model, run_dir / "adapter")
        extra = run_dir / "extra_trainable.pt"
        if extra.exists():
            n = load_extra_trainable(model, torch.load(extra, map_location="cpu"))
            print(f"đã nạp lại {n} tensor projector từ {extra}")
    model.eval()
    return model, processor


@torch.no_grad()
def predict(model, processor, entries, data_dir, coord="norm1000", batch_size=4, max_new_tokens=256):
    prompt = get_prompt(coord)
    rows = []
    for i in range(0, len(entries), batch_size):
        chunk = entries[i:i + batch_size]
        convs = [[{"role": "user", "content": [
            {"type": "image", "image": Image.open(Path(data_dir) / e["image_path"]).convert("RGB")},
            {"type": "text", "text": prompt}]}] for e in chunk]
        inputs = processor.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"image_id": e["image_id"], "raw": t} for e, t in zip(chunk, texts)]
        print(f"{len(rows)}/{len(entries)}", end="\r")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", help="thư mục chứa adapter/ và extra_trainable.pt; bỏ trống = E0")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, help="chỉ chạy N ảnh đầu (gỡ lỗi trên val)")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    entries = read_jsonl(Path(a.data_dir) / "splits" / f"{a.split}.jsonl")
    if a.limit:
        entries = entries[:a.limit]
    model, processor = load_for_inference(a.run_dir)
    rows = predict(model, processor, entries, a.data_dir, a.coord, a.batch_size, a.max_new_tokens)
    write_jsonl(a.out, rows)
    print(f"\nghi {len(rows)} dòng -> {a.out}")
    print(json.dumps(rows[:3], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
