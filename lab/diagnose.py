"""In kiểu module và dtype của các khối visual sau khi nạp 4-bit (để gỡ lỗi lượng tử hoá, GUIDE bước 3, 5).

python -m lab.diagnose
"""
from collections import Counter

from lab.infer import MODEL_ID
from lab.model import load_model


def main():
    model = load_model(MODEL_ID)
    leaf_types = Counter()
    for name, mod in model.named_modules():
        if "visual" in name and not list(mod.children()):
            group = "projector" if ("merger" in name) else "vit"
            leaf_types[(group, type(mod).__name__)] += 1
    print("leaf module types:", dict(leaf_types))
    groups = Counter()
    for name, p in model.named_parameters():
        if "visual" not in name:
            continue
        group = "projector" if "merger" in name else "vit"
        groups[(group, str(p.dtype), type(p).__name__)] += p.numel()
    for k, v in sorted(groups.items()):
        print(k, f"{v:,}")
    print("ví dụ tên projector:")
    for name, p in list(model.named_parameters()):
        if "merger" in name:
            print(" ", name, p.dtype, tuple(p.shape))


if __name__ == "__main__":
    main()
