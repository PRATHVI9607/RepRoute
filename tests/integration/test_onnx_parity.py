"""ONNX parity (PRD §7 rule 4, §12.2 #7): torch vs onnxruntime within 1e-4, payloads bit-exact."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from kai.contracts.windows import WindowSet
from kai.reps.latent import OnnxLatent, OnnxVQ
from kai.tasks import train as tr
from kai.tasks.models import LatentEncoder, RawNet, VQEncoder

N = 1024


def models() -> dict[str, nn.Module]:
    torch.manual_seed(0)
    out = {
        "f_raw": RawNet(),
        "r5_encoder": LatentEncoder(N, 32),
        "r6_encoder": nn.Sequential(LatentEncoder(N, 16), nn.Tanh()),
    }
    for m in out.values():  # exercise non-trivial BatchNorm statistics, as after training
        m.train()
        m(torch.randn(16, 1, N) * 0.05)
        m.eval()
    return out


@pytest.mark.parametrize("name", ["f_raw", "r5_encoder", "r6_encoder"])
def test_float_outputs_match_within_tolerance(name: str, ws: WindowSet) -> None:
    model = models()[name]
    x = ws.x[:64].astype(np.float32) / 32768.0
    ref = tr.predict(model, x)
    sess = ort.InferenceSession(tr.export_onnx(model, N), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": x[:, None, :]})[0]
    assert np.abs(ref - got).max() <= 1e-4
    one = sess.run(None, {"x": x[:1, None, :]})[0]  # dynamic batch axis: single window works
    assert np.abs(one - got[:1]).max() <= 1e-5


def test_int8_latent_payloads_are_bit_exact(ws: WindowSet) -> None:
    model = models()["r6_encoder"]
    rep = OnnxLatent("r6_bottleneck", N, tr.export_onnx(model, N), np.full(16, 1 / 127.0))
    x = ws.x[:128]
    torch_latent = tr.predict(model, x.astype(np.float32) / 32768.0)
    for i, w in enumerate(x):
        payload = rep.encode(w)
        assert len(payload) == 16 == rep.size_model().mean_bytes
        assert payload == rep.quantise(torch_latent[i]).tobytes()
        assert np.allclose(rep.decode_for_head(payload), torch_latent[i], atol=0.5 / 127 + 1e-4)


def test_vq_indices_are_bit_exact_and_decode_to_codebook_rows(ws: WindowSet) -> None:
    torch.manual_seed(1)
    enc = VQEncoder(code_dim=8).eval()
    rep = OnnxVQ("r7_vq", N, tr.export_onnx(enc, N), enc.codebook.detach().numpy())
    x = ws.x[:128]
    with torch.no_grad():
        ref = enc(tr.to_tensor(x)).numpy()
    assert rep.size_model().mean_bytes == N // 256
    for i, w in enumerate(x):
        payload = rep.encode(w)
        assert np.array_equal(np.frombuffer(payload, np.uint8), ref[i])
        assert rep.decode_for_head(payload).shape == (N // 256 * 8,)
    with pytest.raises(ValueError, match="one byte"):
        OnnxVQ("r7_vq", N, rep.onnx_bytes, np.zeros((300, 8), np.float32))


def test_device_resolution() -> None:
    assert tr.resolve_device("cpu").type == "cpu"
    assert tr.resolve_device("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            tr.resolve_device("cuda")
