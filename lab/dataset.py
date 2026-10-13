"""Ví dụ huấn luyện: chat template + mask loss + collate (GUIDE bước 4)."""
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from lab.labels import get_prompt, objects_to_answer
from scripts.labcommon import read_jsonl

SEQ_PAD = {"input_ids": None, "attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}
CAT_KEYS = ("pixel_values", "image_grid_thw")
IMAGE_TOKEN = "<|image_pad|>"


def build_user(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer):
    """Token hoá cả hội thoại; labels = -100 ở phần prompt (gồm token ảnh), chỉ học câu trả lời."""
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    user = build_user(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: mask loss sẽ sai")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100
    full_enc["labels"] = labels
    return full_enc


def collate(batch, pad_id):
    """Pad các tensor chuỗi; nối (cat) pixel_values và image_grid_thw, không pad."""
    length = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for key, value in SEQ_PAD.items():
        if key not in batch[0]:
            continue
        fill = pad_id if value is None else value
        out[key] = torch.stack([F.pad(b[key][0], (0, length - b[key].shape[1]), value=fill)
                                for b in batch])
    for key in CAT_KEYS:
        out[key] = torch.cat([b[key] for b in batch], dim=0)
    return out


def describe_example(processor, enc):
    """Số token ảnh và phần văn bản được tính loss (để kiểm tra bước 4)."""
    tok = processor.tokenizer
    img_id = tok.convert_tokens_to_ids(IMAGE_TOKEN)
    ids, labels = enc["input_ids"][0], enc["labels"][0]
    return {"n_image_tokens": int((ids == img_id).sum()),
            "n_supervised": int((labels != -100).sum()),
            "supervised_text": tok.decode(ids[labels != -100], skip_special_tokens=False)}


def subset(entries, fraction, seed):
    if fraction >= 1.0:
        return list(entries)
    k = max(1, round(len(entries) * fraction))
    rng = random.Random(seed)
    return [entries[i] for i in sorted(rng.sample(range(len(entries)), k))]


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, split, processor, data_dir="data", coord="norm1000", fraction=1.0, seed=42):
        from PIL import Image
        self._open = Image.open
        self.data_dir = Path(data_dir)
        self.processor, self.coord = processor, coord
        self.prompt = get_prompt(coord)
        self.entries = subset(read_jsonl(self.data_dir / "splits" / f"{split}.jsonl"), fraction, seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        image = self._open(self.data_dir / e["image_path"]).convert("RGB")
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.processor, image, self.prompt, answer)
