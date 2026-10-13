"""Chấm một lần chạy bằng scripts/score.py, tự lấy --meta từ run.json.

python -m lab.score_run --name E1 --pred work/e1.jsonl --out report/evidence/e1.json --run-dir <dir>
"""
import argparse
import json
from pathlib import Path

from scripts import score


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", help="có run.json; bỏ trống cho E0")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args(argv)

    args = ["--pred", a.pred, "--name", a.name, "--out", a.out, "--coord", a.coord, "--split", a.split]
    if a.run_dir:
        run = json.loads((Path(a.run_dir) / "run.json").read_text(encoding="utf-8"))
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            args += ["--meta", f"{k}={run[k]}"]
    score.main(args)


if __name__ == "__main__":
    main()
