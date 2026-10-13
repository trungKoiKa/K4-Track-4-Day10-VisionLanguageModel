"""Nạp model 4-bit và setup LoRA / đóng băng cho E1–E3 (GUIDE bước 3, 5)."""
import re

import torch

LLM_RE = (r".*language_model\.layers\.\d+\."
          r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
VIT_RE = (r".*visual\.blocks\.\d+\."
          r"(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))")
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
# transformers 5.x so khớp bằng re.match neo ở đầu tên đầy đủ ("model.visual.…"), nên "visual" trơn
# KHÔNG khớp và ViT bị lượng tử hoá 4-bit. Phải ghi "model.visual".
SKIP_QUANT = ["model.visual", "lm_head"]
EXPECTED_MODULES = {"llm": 252, "llm+vit": 348}


def target_regex(targets):
    if targets == "llm":
        return LLM_RE
    if targets == "llm+vit":
        return f"(?:{LLM_RE})|(?:{VIT_RE})"
    raise ValueError(f"targets không hợp lệ: {targets}")


def matched_names(names, targets):
    rx = re.compile(target_regex(targets))
    return [n for n in names if rx.fullmatch(n)]


def load_processor(model_id):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return processor


def load_model(model_id):
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                               llm_int8_skip_modules=SKIP_QUANT)
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})


def projector_params(model):
    return [(n, p) for n, p in model.named_parameters() if any(k in n for k in PROJECTOR_KEYS)]


def apply_setup(model, targets, r=16, alpha=32, dropout=0.05, train_projector=False):
    """Đóng băng toàn bộ, gắn LoRA vào `targets`, tuỳ chọn mở đầy đủ projector (float32)."""
    from peft import LoraConfig, get_peft_model
    names = [n for n, _ in model.named_modules()]
    matched = matched_names(names, targets)
    if len(matched) != EXPECTED_MODULES[targets]:
        raise RuntimeError(f"LoRA khớp {len(matched)} module, cần {EXPECTED_MODULES[targets]}: "
                           "in model.named_modules() để chỉnh regex")
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=target_regex(targets)))
    if train_projector:
        for _, p in projector_params(model):
            p.requires_grad = True
    # GradScaler không unscale được gradient fp16: mọi tham số huấn luyện phải ở float32
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_n = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    proj_n = sum(p.numel() for n, p in projector_params(model) if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    print(f"LoRA modules: {len(matched)} | tham số huấn luyện: {trainable:,} "
          f"(LoRA {lora_n:,} + projector {proj_n:,}) | dtype: {dtypes}")
    return model, {"lora_modules": len(matched), "trainable_params": trainable,
                   "lora_params": lora_n, "projector_params": proj_n}


def extra_trainable_state(model):
    """Tham số huấn luyện đầy đủ ngoài LoRA (projector). peft.save_pretrained KHÔNG lưu chúng."""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad and "lora_" not in n}


def load_extra_trainable(model, state):
    """Nạp lại projector đã huấn luyện (khớp tên sau khi bỏ tiền tố peft)."""
    def norm(n):
        return n.replace("base_model.model.", "", 1)
    params = {norm(n): p for n, p in model.named_parameters()}
    for n, v in state.items():
        p = params[norm(n)]
        p.data = p.data.float()
        p.data.copy_(v.to(p.device))
    return len(state)
