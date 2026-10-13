import json

from lab.analyze import main, misread_scale, split_stats
from lab.labels import objects_to_answer
from scripts.labcommon import write_jsonl


def make_data(tmp_path):
    objs = [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]
    for s, sess in (("train", "a"), ("val", "b"), ("test", "c")):
        write_jsonl(tmp_path / "splits" / f"{s}.jsonl", [
            {"image_id": f"{s}1", "image_path": "x.jpg", "width": 640, "height": 360,
             "session": sess, "objects": objs}])
    (tmp_path / "splits" / "meta.json").write_text(json.dumps({"dataset_version": "t"}), encoding="utf-8")
    return objs


def test_split_stats_counts_sessions_and_median_area(tmp_path):
    make_data(tmp_path)
    st = split_stats(tmp_path)
    assert st["train"]["sessions"] == 1 and st["test"]["boxes"]["pothole"] == 1
    assert st["all"]["median_area_pct"]["pothole"] == 4.0


def test_perfect_prediction_collapses_when_read_as_pixels(tmp_path):
    objs = make_data(tmp_path)
    write_jsonl(tmp_path / "p.jsonl", [{"image_id": "test1", "raw": objects_to_answer(objs)}])
    r = misread_scale(tmp_path, tmp_path / "p.jsonl")
    assert r["norm1000"]["f1"] == 1.0
    assert r["pixel"]["f1"] < 0.01


def test_cli_writes_json(tmp_path):
    objs = make_data(tmp_path)
    write_jsonl(tmp_path / "p.jsonl", [{"image_id": "test1", "raw": objects_to_answer(objs)}])
    main(["--pred", str(tmp_path / "p.jsonl"), "--data-dir", str(tmp_path), "--out", str(tmp_path / "o.json")])
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["scale_misread"]
