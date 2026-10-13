import json

import pytest

from lab.labels import box_to_pixels, get_prompt, objects_to_answer, yolo_to_box
from lab.reward import reward
from scripts.score import iou_xyxy, parse_output


def test_yolo_to_box_norm1000_known_values():
    b = yolo_to_box("pothole", (0.5, 0.5, 0.2, 0.4))
    assert b == {"label": "pothole", "bbox_2d": [400, 300, 600, 700]}


def test_yolo_to_box_pixel_and_clamp():
    b = yolo_to_box("crack", (0.5, 0.5, 0.2, 0.4), coord="pixel", width=640, height=360)
    assert b["bbox_2d"] == [256, 108, 384, 252]
    edge = yolo_to_box("crack", (0.99, 0.01, 0.1, 0.1))
    assert all(0 <= v <= 1000 for v in edge["bbox_2d"])


def test_roundtrip_back_to_pixels_matches_original():
    yolo = (0.3125, 0.5, 0.125, 0.25)          # = pixel [160, 135, 240, 225] trên 640x360
    b = yolo_to_box("manhole", yolo)
    x1, y1, x2, y2 = box_to_pixels(b["bbox_2d"], "norm1000", 640, 360)
    assert (x1, y1, x2, y2) == pytest.approx((160, 135, 240, 225), abs=0.5)


def test_answer_sorted_by_y_then_x_and_compact():
    objs = [{"label": "crack", "yolo": (0.5, 0.8, 0.1, 0.1)},
            {"label": "pothole", "yolo": (0.7, 0.2, 0.1, 0.1)},
            {"label": "manhole", "yolo": (0.2, 0.2, 0.1, 0.1)}]
    ans = objects_to_answer(objs)
    labels = [o["label"] for o in json.loads(ans)]
    assert labels == ["manhole", "pothole", "crack"]
    assert " " not in ans
    assert objects_to_answer([]) == "[]"
    assert parse_output(ans) is not None


def test_prompt_states_scale_and_classes():
    p = get_prompt("norm1000")
    assert "0–1000" in p and all(c in p for c in ("pothole", "crack", "manhole"))
    assert "pixel" in get_prompt("pixel")


GT = [{"label": "pothole", "bbox": (100, 100, 300, 300)}]
GOOD = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 300, 300]}])


def test_reward_perfect_is_one():
    assert reward(GOOD, GT) == pytest.approx(1.0)


def test_reward_broken_json_is_zero():
    assert reward("không phải json", GT) == 0.0


def test_reward_wrong_class_gets_only_format_credit():
    wrong = json.dumps([{"label": "crack", "bbox_2d": [100, 100, 300, 300]}])
    assert reward(wrong, GT) == pytest.approx(0.2)


def test_reward_extra_box_is_penalised():
    extra = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 300, 300]},
                        {"label": "crack", "bbox_2d": [500, 500, 600, 600]}])
    assert reward(extra, GT) == pytest.approx(0.2 + 0.8 * 1.0 / 2)


def test_reward_empty_cases():
    assert reward("[]", []) == 1.0
    assert reward("[]", GT) == pytest.approx(0.2)
    assert iou_xyxy((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
