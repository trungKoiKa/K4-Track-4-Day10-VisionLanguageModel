#!/bin/bash
# Chạy trọn một thí nghiệm: huấn luyện -> suy luận test -> chấm -> sao lưu lên Drive.
# Dùng: bash lab/run_exp.sh <tên> <config> [--set khoá=giá_trị ...] [-- coord=pixel]
#   bash lab/run_exp.sh e1 e1
#   bash lab/run_exp.sh e4_rank_4 e3 --set lora.r=4 --set lora.alpha=8
set -e
NAME=$1
CFG=$2
shift 2
OUT=${OUT:-/content/drive/MyDrive/lab10}
COORD=norm1000
for a in "$@"; do
  case "$a" in coord_scale=pixel) COORD=pixel ;; esac
done
UP=$(echo "$NAME" | tr 'a-z' 'A-Z')

python -m lab.train --config "configs/$CFG.yaml" --out "$OUT/$NAME" "$@"
python -m lab.infer --run-dir "$OUT/$NAME" --coord "$COORD" --out "work/$NAME.jsonl"
mkdir -p "$OUT/preds" "$OUT/evidence"
cp "work/$NAME.jsonl" "$OUT/preds/$NAME.jsonl"
python -m lab.score_run --name "$UP" --pred "work/$NAME.jsonl" --run-dir "$OUT/$NAME" \
  --coord "$COORD" --out "report/evidence/$NAME.json"
cp "report/evidence/$NAME.json" "$OUT/evidence/$NAME.json"
echo "XONG $NAME"
