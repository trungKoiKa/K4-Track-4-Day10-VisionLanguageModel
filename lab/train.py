"""Vòng huấn luyện QLoRA (GUIDE bước 6).

python -m lab.train --config configs/e1.yaml --out /content/drive/MyDrive/lab10/e1
python -m lab.train --config configs/e3.yaml --out ... --set lora.r=4 --set lora.alpha=8   # E4
"""
import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch

from lab.ckpt import find_checkpoint, load_train_state, save_checkpoint
from lab.config import load_config
from lab.dataset import RoadDataset, collate
from lab.model import (apply_setup, extra_trainable_state, load_extra_trainable, load_model,
                       load_processor)


def export_final(model, out):
    """Bản cuối cùng mà lab.infer đọc: <out>/adapter và <out>/extra_trainable.pt."""
    out = Path(out)
    model.save_pretrained(out / "adapter")
    extra = extra_trainable_state(model)
    if extra:
        torch.save(extra, out / "extra_trainable.pt")


def restore_weights(model, ckpt_dir):
    from peft import load_peft_weights, set_peft_model_state_dict
    ckpt_dir = Path(ckpt_dir)
    set_peft_model_state_dict(model, load_peft_weights(str(ckpt_dir / "adapter")))
    extra = ckpt_dir / "extra_trainable.pt"
    if extra.exists():
        load_extra_trainable(model, torch.load(extra, map_location="cpu"))


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--set", action="append", default=[], help="khoá.con=giá_trị")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=100, help="lưu checkpoint mỗi N bước")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ <out>/ckpt nếu có")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    processor = load_processor(cfg["model_id"])
    model = load_model(cfg["model_id"])
    lora = cfg["lora"]
    model, info = apply_setup(model, lora["targets"], lora["r"], lora["alpha"], lora["dropout"],
                              cfg["train_projector"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    ds = RoadDataset("train", processor, a.data_dir, cfg["coord_scale"], cfg["data_size"], seed)
    print(f"{cfg['name']}: {len(ds)} ảnh train, coord={cfg['coord_scale']}")
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=a.workers,
        collate_fn=partial(collate, pad_id=processor.tokenizer.pad_token_id), generator=gen)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    total, accum = cfg["max_steps"], cfg["grad_accum"]
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, warmup, total))
    scaler = torch.amp.GradScaler("cuda")

    start, log, elapsed0 = 0, [], 0.0
    ckpt = find_checkpoint(out) if a.resume else None
    if ckpt:
        restore_weights(model, ckpt)
        st = load_train_state(ckpt, opt, sched, scaler)
        start, log, elapsed0 = st["step"], st["log"], st["elapsed_min"]
        gen.manual_seed(seed + start)   # thứ tự batch sau resume khác lần chạy liền mạch (chấp nhận được)
        print(f"RESUME từ bước {start}/{total} (đã train {elapsed0:.1f} phút)")
    elif a.resume:
        print("--resume: chưa có checkpoint, train từ đầu")

    def batches():
        while True:
            yield from loader
    stream = batches()
    t0 = time.time() - elapsed0 * 60
    for step in range(start + 1, total + 1):
        step_loss = 0.0
        for _ in range(accum):
            batch = {k: v.to(model.device) for k, v in next(stream).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: nguy cơ tràn số fp16")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        log.append({"step": step, "loss": step_loss, "lr": sched.get_last_lr()[0]})
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{total} loss {step_loss:.4f} lr {log[-1]['lr']:.2e} "
                  f"{(time.time() - t0) / 60:.1f} phút")
        if step % a.save_every == 0 or step == total:
            save_checkpoint(out, model, extra_trainable_state(model), step, opt, sched, scaler, log,
                            (time.time() - t0) / 60)
    export_final(model, out)

    run = {"name": cfg["name"], "config": cfg, "lora_modules": info["lora_modules"],
           "trainable_params": info["trainable_params"],
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
           "train_minutes": round((time.time() - t0) / 60, 2),
           "n_train_images": len(ds), "final_loss": log[-1]["loss"], "loss_log": log}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes", "final_loss")})


if __name__ == "__main__":
    main()
