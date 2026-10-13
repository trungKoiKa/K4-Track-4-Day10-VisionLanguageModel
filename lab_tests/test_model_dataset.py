import json

import pytest
import torch

from lab.config import load_config
from lab.dataset import collate, describe_example, encode_example, subset
from lab.labels import get_prompt, objects_to_answer
from lab.model import EXPECTED_MODULES, matched_names
from lab.train import cosine_with_warmup


def fake_names():
    names = ["model.visual.merger.linear_fc1", "model.visual.deepstack_merger_list.0.linear_fc1",
             "model.visual.blocks.0.attn.qkv_extra", "lm_head"]
    for n in range(36):
        base = f"base_model.model.model.language_model.layers.{n}"
        names += [f"{base}.self_attn.{x}_proj" for x in "qkvo"]
        names += [f"{base}.mlp.{x}_proj" for x in ("gate", "up", "down")]
        names += [f"{base}.self_attn.q_norm", f"{base}.input_layernorm"]
    for n in range(24):
        base = f"model.visual.blocks.{n}"
        names += [f"{base}.attn.qkv", f"{base}.attn.proj",
                  f"{base}.mlp.linear_fc1", f"{base}.mlp.linear_fc2", f"{base}.norm1"]
    return names


def test_lora_target_counts_match_guide():
    names = fake_names()
    assert len(matched_names(names, "llm")) == EXPECTED_MODULES["llm"] == 252
    assert len(matched_names(names, "llm+vit")) == EXPECTED_MODULES["llm+vit"] == 348
    assert not any("merger" in n for n in matched_names(names, "llm+vit"))


def test_skip_quant_keeps_vit_projector_and_lm_head_unquantized():
    from transformers.quantizers.quantizers_utils import should_convert_module
    from lab.model import SKIP_QUANT
    for name in ("model.visual.merger.linear_fc1", "model.visual.deepstack_merger_list.0.linear_fc2",
                 "model.visual.blocks.3.attn.qkv", "lm_head"):
        assert not should_convert_module(name, SKIP_QUANT), name
    assert should_convert_module("model.language_model.layers.0.self_attn.q_proj", SKIP_QUANT)
    # mẫu "visual" trơn của GUIDE không bắt được tên đầy đủ trong transformers 5.19
    assert should_convert_module("model.visual.merger.linear_fc1", ["visual", "lm_head"])


def test_collate_pads_sequences_and_concats_vision():
    def ex(n, n_img):
        return {"input_ids": torch.arange(1, n + 1)[None], "attention_mask": torch.ones(1, n, dtype=torch.long),
                "mm_token_type_ids": torch.zeros(1, n, dtype=torch.long),
                "labels": torch.arange(1, n + 1)[None],
                "pixel_values": torch.zeros(n_img, 8), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    out = collate([ex(5, 4), ex(3, 4)], pad_id=99)
    assert out["input_ids"].shape == (2, 5)
    assert out["input_ids"][1].tolist() == [1, 2, 3, 99, 99]
    assert out["labels"][1].tolist() == [1, 2, 3, -100, -100]
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 0, 0]
    assert out["pixel_values"].shape == (8, 8) and out["image_grid_thw"].shape == (2, 3)


def test_subset_is_deterministic_and_sized():
    items = list(range(100))
    a, b = subset(items, 0.1, 42), subset(items, 0.1, 42)
    assert a == b and len(a) == 10 and subset(items, 1.0, 42) == items


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: X\nlora: {r: 16, alpha: 32}\n", encoding="utf-8")
    cfg = load_config(p, ["lora.r=4", "lora.alpha=8", "data_size=0.1", "coord_scale=pixel"])
    assert cfg["lora"] == {"r": 4, "alpha": 8} and cfg["data_size"] == 0.1 and cfg["coord_scale"] == "pixel"


def test_lr_schedule_warmup_then_cosine_to_zero():
    assert cosine_with_warmup(0, 15, 300) == pytest.approx(1 / 15)
    assert cosine_with_warmup(15, 15, 300) == pytest.approx(1.0)
    assert cosine_with_warmup(299, 15, 300) < 0.01


@pytest.fixture(scope="module")
def processor():
    try:
        from lab.model import load_processor
        return load_processor("Qwen/Qwen3-VL-4B-Instruct")
    except Exception as e:  # không có mạng / chưa tải được
        pytest.skip(f"không nạp được processor: {e}")


def test_encode_example_masks_prompt_and_image_tokens(processor):
    from PIL import Image
    objs = [{"label": "pothole", "yolo": (0.5, 0.5, 0.2, 0.2)}]
    answer = objects_to_answer(objs)
    enc = encode_example(processor, Image.new("RGB", (640, 360)), get_prompt(), answer)
    info = describe_example(processor, enc)
    assert info["n_image_tokens"] == 220
    assert info["supervised_text"].startswith(answer)
    assert info["supervised_text"].lstrip().startswith("[{")
    assert "<|im_end|>" in info["supervised_text"]
    assert json.loads(info["supervised_text"].split("<|im_end|>")[0]) == json.loads(answer)
