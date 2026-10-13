import torch

from lab.ckpt import CKPT, find_checkpoint, load_train_state, save_checkpoint
from lab.train import cosine_with_warmup


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(4, 2)

    def save_pretrained(self, path):   # giả lập peft
        path.mkdir(parents=True)
        torch.save(self.state_dict(), path / "adapter_model.bin")


def make(total=20, warmup=2):
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, warmup, total))
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, opt, sched, scaler


def train_steps(model, opt, sched, n):
    for _ in range(n):
        opt.zero_grad()
        model.w(torch.ones(3, 4)).sum().backward()
        opt.step()
        sched.step()


def test_resume_matches_uninterrupted_run(tmp_path):
    torch.manual_seed(0)
    a, opt_a, sch_a, _ = make()
    train_steps(a, opt_a, sch_a, 20)                       # chạy liền mạch 20 bước

    torch.manual_seed(0)
    b, opt_b, sch_b, sc_b = make()
    train_steps(b, opt_b, sch_b, 10)
    save_checkpoint(tmp_path, b, {}, 10, opt_b, sch_b, sc_b, [{"step": 10}], 1.5)

    c, opt_c, sch_c, sc_c = make()                         # "tiến trình mới" sau khi bị ngắt
    ck = find_checkpoint(tmp_path)
    c.load_state_dict(torch.load(ck / "adapter" / "adapter_model.bin"))
    st = load_train_state(ck, opt_c, sch_c, sc_c)
    assert st["step"] == 10 and st["elapsed_min"] == 1.5 and st["log"] == [{"step": 10}]
    train_steps(c, opt_c, sch_c, 10)

    for p, q in zip(a.parameters(), c.parameters()):
        assert torch.allclose(p, q, atol=1e-6)
    assert sch_a.get_last_lr() == sch_c.get_last_lr()


def test_interrupted_write_keeps_previous_checkpoint(tmp_path):
    m, opt, sch, sc = make()
    save_checkpoint(tmp_path, m, {}, 100, opt, sch, sc, [], 1.0)
    (tmp_path / (CKPT + ".tmp")).mkdir()                   # mô phỏng bị ngắt khi đang ghi bản mới
    (tmp_path / (CKPT + ".tmp") / "train_state.pt").write_bytes(b"hong")
    ck = find_checkpoint(tmp_path)
    assert ck == tmp_path / CKPT
    assert torch.load(ck / "train_state.pt", weights_only=False)["step"] == 100


def test_second_save_replaces_first(tmp_path):
    m, opt, sch, sc = make()
    save_checkpoint(tmp_path, m, {}, 100, opt, sch, sc, [], 1.0)
    save_checkpoint(tmp_path, m, {}, 200, opt, sch, sc, [], 2.0)
    assert torch.load(tmp_path / CKPT / "train_state.pt", weights_only=False)["step"] == 200
    assert not (tmp_path / (CKPT + ".old")).exists()


def test_no_checkpoint_returns_none(tmp_path):
    assert find_checkpoint(tmp_path) is None
