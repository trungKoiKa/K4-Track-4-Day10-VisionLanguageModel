"""Checkpoint có thể resume: trọng số + trạng thái huấn luyện, ghi nguyên tử (GUIDE bước 6).

Một checkpoint = thư mục `<out>/ckpt/` gồm:
    adapter/            LoRA (peft save_pretrained)
    extra_trainable.pt  projector (E2, E3)
    train_state.pt      step, optimizer, scheduler, scaler, loss log, thời gian đã train
Cả thư mục được ghi vào `ckpt.tmp/` rồi đổi tên một lần, nên ngắt giữa chừng không làm hỏng checkpoint cũ.
"""
import os
import shutil
from pathlib import Path

import torch

CKPT = "ckpt"


def swap_dir(tmp, final):
    """Thay `final` bằng `tmp`; bản cũ chỉ bị xoá sau khi bản mới đã vào đúng chỗ."""
    tmp, final = Path(tmp), Path(final)
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)


def save_checkpoint(out, model, extra, step, opt, sched, scaler, log, elapsed_min):
    out = Path(out)
    tmp = out / (CKPT + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    if extra:
        torch.save(extra, tmp / "extra_trainable.pt")
    torch.save({"step": step, "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "log": log, "elapsed_min": elapsed_min},
               tmp / "train_state.pt")
    swap_dir(tmp, out / CKPT)


def find_checkpoint(out):
    d = Path(out) / CKPT
    return d if (d / "train_state.pt").exists() else None


def load_train_state(ckpt_dir, opt, sched, scaler):
    st = torch.load(Path(ckpt_dir) / "train_state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    scaler.load_state_dict(st["scaler"])
    return st
