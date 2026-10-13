"""Nhãn YOLO -> JSON bbox_2d và prompt (GUIDE bước 2)."""
import json

from scripts.labcommon import CLASSES

COORDS = ("norm1000", "pixel")

_HEAD = ("Bạn là hệ thống phát hiện hư hại mặt đường từ ảnh dashcam. "
         "Hãy tìm mọi đối tượng thuộc 3 lớp: pothole (ổ gà), crack (vết nứt), manhole (nắp cống). ")
_FORMAT = ('Chỉ trả về MỘT JSON list, không thêm chữ nào khác, mỗi phần tử có dạng '
           '{"label": "<lớp>", "bbox_2d": [x1, y1, x2, y2]}. Nếu không có đối tượng nào, trả về []. ')
_SCALE = {
    "norm1000": "Toạ độ là số nguyên trên thang 0–1000 (tương đối so với chiều rộng và chiều cao ảnh), "
                "gốc ở góc trên bên trái.",
    "pixel": "Toạ độ là số nguyên tính bằng pixel của ảnh, gốc ở góc trên bên trái.",
}


def get_prompt(coord="norm1000"):
    if coord not in COORDS:
        raise ValueError(f"coord phải thuộc {COORDS}")
    return _HEAD + _FORMAT + _SCALE[coord]


def _clamp_round(v, hi):
    return int(min(max(round(v), 0), hi))


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """(xc, yc, w, h) chuẩn hoá [0,1] -> {"label", "bbox_2d"} số nguyên, kẹp vào khung."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    x1, y1 = (xc - w / 2) * sx, (yc - h / 2) * sy
    x2, y2 = (xc + w / 2) * sx, (yc + h / 2) * sy
    return {"label": label,
            "bbox_2d": [_clamp_round(x1, sx), _clamp_round(y1, sy),
                        _clamp_round(x2, sx), _clamp_round(y2, sy)]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    """Danh sách object của một ảnh -> chuỗi JSON, box sắp theo (y1, x1)."""
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(bbox, coord="norm1000", width=640, height=360):
    """Quy bbox_2d về pixel (float) để vẽ lại lên ảnh."""
    if coord == "pixel":
        return tuple(float(v) for v in bbox)
    x1, y1, x2, y2 = bbox
    return (x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height)


def check_labels(label):
    return label in CLASSES
