"""Dump per-primitive parity fixtures (RoPE, SSMax, MHA) from the reference code.

Each fixture is a self-contained random-weight module: we build the real Python
module, randomize ALL its parameters (so the test exercises real arithmetic, not
zero-init shortcuts), run a forward pass on random inputs, and save weights +
inputs + the golden output as one safetensors file. The R port loads the
weights, runs its own forward, and must match the golden output (atol ~1e-5).

Run after installing the venv. Usage:
    python dump_primitives.py

Output: tests/testthat/fixtures/tabicl/<name>.safetensors  (+ <name>.json meta)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save

from tabicl._model.rope import RotaryEmbedding
from tabicl._model.ssmax import QASSMaxMLP
from tabicl._model.layers import MultiheadAttention
from tabicl._model.interaction import RowInteraction
from tabicl._model.embedding import ColEmbedding
from tabicl._model.learning import ICLearning
from tabicl._model.tabicl import TabICL

OUT = Path(__file__).resolve().parents[2] / "tests" / "testthat" / "fixtures" / "tabicl"
SEED = int(hashlib.sha256(b"brulee-tabicl-primitives-v1").hexdigest(), 16) % (2**31 - 1)


def randomize_(module: torch.nn.Module, gen: torch.Generator) -> None:
    """Fill every parameter with non-trivial random values (in place).

    Uses fan-in (Xavier-style) scaling for weight matrices and a small constant
    for vectors so activations stay O(1) through the stack. That keeps the
    fixtures in-distribution (no softmax saturation, no exploding outputs), so a
    tight absolute tolerance in the R parity tests is meaningful.
    """
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() >= 2:
                std = 1.0 / (p.shape[-1] ** 0.5)
            else:
                std = 0.1
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * std)


def write(name: str, tensors: dict, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Gzip the safetensors blob. The raw safetensors header is a u64 length
    # prefix whose leading byte can coincide with an executable opcode (e.g.
    # 0xb8), which R CMD check flags as a bundled executable. gzip's magic bytes
    # avoid that for every fixture; the R loader decompresses on read.
    blob = save({k: v.contiguous().to(torch.float32) for k, v in tensors.items()})
    gz_path = OUT / f"{name}.safetensors.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(blob)
    json_path = OUT / f"{name}.json"
    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    os.chmod(gz_path, 0o644)
    os.chmod(json_path, 0o644)
    shapes = {k: list(v.shape) for k, v in tensors.items()}
    print(f"{name}: {shapes}")


def dump_rope(gen):
    # Row-interactor RoPE: non-interleaved, theta=100000. Small head_dim keeps
    # the fixture tiny; the rotation is dimension-agnostic.
    head_dim, num_heads, seq = 8, 8, 7
    rope = RotaryEmbedding(dim=head_dim, interleaved=False, theta=100000)
    rope.eval()
    with torch.no_grad():
        rope.freqs.copy_(torch.randn(rope.freqs.shape, generator=gen) * 0.1 + 0.5)
    q = torch.randn(2, num_heads, seq, head_dim, generator=gen)
    with torch.no_grad():
        out = rope.rotate_queries_or_keys(q)
    write(
        "rope",
        {"freqs": rope.freqs.detach(), "q": q, "out": out},
        {"head_dim": head_dim, "num_heads": num_heads, "seq": seq,
         "interleaved": False, "theta": 100000, "seed": SEED},
    )


def dump_ssmax(gen):
    # qassmax-mlp-elementwise as used by col attn1 and icl blocks.
    num_heads, head_dim, n_hidden, seq, n = 8, 8, 64, 6, 11
    ss = QASSMaxMLP(num_heads, head_dim, n_hidden=n_hidden, elementwise=True)
    ss.eval()
    randomize_(ss, gen)  # also un-zeros query_mlp so modulation != 1
    q = torch.randn(3, num_heads, seq, head_dim, generator=gen)
    with torch.no_grad():
        out = ss(q, n)
    sd = {f"ssmax.{k}": v.detach() for k, v in ss.state_dict().items()}
    write(
        "ssmax",
        {**sd, "q": q, "out": out},
        {"num_heads": num_heads, "head_dim": head_dim, "n_hidden": n_hidden,
         "seq": seq, "n": n, "elementwise": True, "seed": SEED},
    )


def dump_mha(name, embed_dim, num_heads, use_rope, ssmax, q_len, kv_len, gen):
    mha = MultiheadAttention(embed_dim, num_heads, dropout=0.0, ssmax=ssmax)
    mha.eval()
    randomize_(mha, gen)

    rope = None
    extra = {}
    if use_rope:
        head_dim = embed_dim // num_heads
        rope = RotaryEmbedding(dim=head_dim, interleaved=False, theta=100000)
        rope.eval()
        with torch.no_grad():
            rope.freqs.copy_(torch.randn(rope.freqs.shape, generator=gen) * 0.1 + 0.5)
        extra["rope.freqs"] = rope.freqs.detach()

    query = torch.randn(2, q_len, embed_dim, generator=gen)
    if q_len == kv_len:
        key = value = query  # self-attention
    else:
        key = torch.randn(2, kv_len, embed_dim, generator=gen)
        value = key  # cross-attention: key is value (as in ISAB)

    with torch.no_grad():
        out = mha(query, key, value, rope=rope)

    tensors = {
        "in_proj_weight": mha.in_proj_weight.detach(),
        "in_proj_bias": mha.in_proj_bias.detach(),
        "out_proj.weight": mha.out_proj.weight.detach(),
        "out_proj.bias": mha.out_proj.bias.detach(),
        "query": query.clone(),
        "key": key.clone(),
        "value": value.clone(),
        "out": out,
        **extra,
    }
    if mha.ssmax_layer is not None:
        for k, v in mha.ssmax_layer.state_dict().items():
            tensors[f"ssmax.{k}"] = v.detach()

    write(
        name,
        tensors,
        {"embed_dim": embed_dim, "num_heads": num_heads, "use_rope": use_rope,
         "ssmax": ssmax, "q_len": q_len, "kv_len": kv_len, "seed": SEED},
    )


def dump_row_interaction(gen, name="row_interaction", bias_free_ln=False):
    # Stage-2 RowInteraction, shrunk: embed_dim 32, 3 blocks, 4 heads, 2 CLS.
    # num_blocks=3 exercises both the self-attention blocks and the final
    # CLS-as-query cross-attention block. bias_free_ln toggles LayerNorm biases
    # (the classifier checkpoint has them, the regressor does not).
    embed_dim, num_blocks, nhead, num_cls = 32, 3, 4, 2
    ff = 2 * embed_dim
    ri = RowInteraction(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        nhead=nhead,
        dim_feedforward=ff,
        num_cls=num_cls,
        rope_base=100000,
        rope_interleaved=False,
        bias_free_ln=bias_free_ln,
    )
    ri.eval()
    randomize_(ri, gen)
    # Give the RoPE frequencies realistic decay rather than tiny random noise.
    with torch.no_grad():
        idx = torch.arange(0, embed_dim // nhead, 2, dtype=torch.float32)
        ri.tf_row.rope.freqs.copy_(1.0 / (100000 ** (idx / (embed_dim // nhead))))

    # col_embed-shaped input: (B, T, H + C, E). The first num_cls slots are
    # placeholders the module overwrites with its CLS tokens.
    b, t, h = 1, 5, 3
    inp = torch.randn(b, t, h + num_cls, embed_dim, generator=gen)
    with torch.no_grad():
        out = ri(inp.clone())  # clone: RowInteraction mutates its input in place

    tensors = {f"ri.{k}": v.detach() for k, v in ri.state_dict().items()}
    tensors["input"] = inp
    tensors["out"] = out
    write(
        name,
        tensors,
        {"embed_dim": embed_dim, "num_blocks": num_blocks, "nhead": nhead,
         "num_cls": num_cls, "dim_feedforward": ff, "rope_base": 100000,
         "bias_free_ln": bias_free_ln, "seed": SEED},
    )


def dump_col_embedding(gen, name, max_classes, bias_free_ln):
    # Stage-1 ColEmbedding in the configuration the v2 checkpoints use:
    # feature_group="same", target_aware, affine=False, qassmax ssmax. Two
    # variants exercise both target encoders: classification (one-hot, max_classes>0)
    # and regression (linear, max_classes=0).
    embed_dim, num_blocks, nhead, num_inds = 32, 2, 4, 8
    feature_group_size, reserve_cls = 3, 2
    ce = ColEmbedding(
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        nhead=nhead,
        dim_feedforward=2 * embed_dim,
        num_inds=num_inds,
        activation="gelu",
        norm_first=True,
        bias_free_ln=bias_free_ln,
        affine=False,
        feature_group="same",
        feature_group_size=feature_group_size,
        target_aware=True,
        max_classes=max_classes,
        reserve_cls_tokens=reserve_cls,
        ssmax="qassmax-mlp-elementwise",
    )
    ce.eval()
    randomize_(ce, gen)

    b, t, h, train_size = 1, 6, 4, 4
    x = torch.randn(b, t, h, generator=gen)
    if max_classes > 0:
        y_train = torch.randint(0, 3, (b, train_size), generator=gen).float()
    else:
        y_train = torch.randn(b, train_size, generator=gen)
    with torch.no_grad():
        out = ce(x, y_train)

    tensors = {f"ce.{k}": v.detach() for k, v in ce.state_dict().items()}
    tensors["X"] = x
    tensors["y_train"] = y_train
    tensors["out"] = out
    write(
        name,
        tensors,
        {"embed_dim": embed_dim, "num_blocks": num_blocks, "nhead": nhead,
         "num_inds": num_inds, "dim_feedforward": 2 * embed_dim,
         "feature_group_size": feature_group_size, "reserve_cls_tokens": reserve_cls,
         "max_classes": max_classes, "bias_free_ln": bias_free_ln,
         "train_size": train_size, "seed": SEED},
    )


def dump_icl_learning(gen, name, max_classes, bias_free_ln):
    # Stage-3 ICLearning. Classification (one-hot encoder, biased LN, out_dim =
    # max_classes) and regression (linear encoder, bias-free LN, out_dim = small
    # quantile count).
    d_model, num_blocks, nhead = 32, 2, 4
    out_dim = max_classes if max_classes > 0 else 16
    icl = ICLearning(
        max_classes=max_classes,
        out_dim=out_dim,
        d_model=d_model,
        num_blocks=num_blocks,
        nhead=nhead,
        dim_feedforward=2 * d_model,
        activation="gelu",
        norm_first=True,
        bias_free_ln=bias_free_ln,
        ssmax="qassmax-mlp-elementwise",
    )
    icl.eval()
    randomize_(icl, gen)

    b, t, train_size = 1, 6, 4
    r = torch.randn(b, t, d_model, generator=gen)
    if max_classes > 0:
        y_train = torch.randint(0, 3, (b, train_size), generator=gen).float()
    else:
        y_train = torch.randn(b, train_size, generator=gen)
    with torch.no_grad():
        out = icl(r.clone(), y_train)  # icl mutates R in place; clone the saved input

    tensors = {f"icl.{k}": v.detach() for k, v in icl.state_dict().items()}
    tensors["R"] = r
    tensors["y_train"] = y_train
    tensors["out"] = out
    write(
        name,
        tensors,
        {"d_model": d_model, "num_blocks": num_blocks, "nhead": nhead,
         "out_dim": out_dim, "max_classes": max_classes,
         "bias_free_ln": bias_free_ln, "train_size": train_size, "seed": SEED},
    )


def _small_tabicl_config(max_classes, bias_free_ln):
    return dict(
        max_classes=max_classes,
        num_quantiles=16,
        embed_dim=16,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=8,
        col_affine=False,
        col_feature_group="same",
        col_feature_group_size=3,
        col_target_aware=True,
        col_ssmax="qassmax-mlp-elementwise",
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=2,
        row_rope_base=100000,
        row_rope_interleaved=False,
        icl_num_blocks=2,
        icl_nhead=4,
        icl_ssmax="qassmax-mlp-elementwise",
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        bias_free_ln=bias_free_ln,
    )


def dump_full_model(gen, name, max_classes, bias_free_ln):
    # End-to-end TabICL forward (col -> row -> icl) on a small config. Validates
    # the full wiring and the weight-key mapping the production loader will use.
    config = _small_tabicl_config(max_classes, bias_free_ln)
    model = TabICL(**config)
    model.eval()
    randomize_(model, gen)

    b, t, h, train_size = 1, 9, 4, 6
    x = torch.randn(b, t, h, generator=gen)
    if max_classes > 0:
        y_train = torch.randint(0, 3, (b, train_size), generator=gen).float()
    else:
        y_train = torch.randn(b, train_size, generator=gen)
    with torch.no_grad():
        out = model(x, y_train)  # return_logits=True default

    # Keep only the three forward stages (drop e.g. quantile_dist params).
    keep = ("col_embedder.", "row_interactor.", "icl_predictor.")
    tensors = {
        k: v.detach()
        for k, v in model.state_dict().items()
        if k.startswith(keep)
    }
    tensors["X"] = x
    tensors["y_train"] = y_train
    tensors["out"] = out
    write(name, tensors, {"config": config, "train_size": train_size, "seed": SEED})


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)

    # embed_dim=64, num_heads=8 (head_dim=8) keeps 8-head behaviour while
    # keeping fixtures small; the ops are dimension-agnostic.
    dump_rope(gen)
    dump_ssmax(gen)
    # Row interactor: self-attn + rope, no ssmax.
    dump_mha("mha_rope", 64, 8, use_rope=True, ssmax="none", q_len=7, kv_len=7, gen=gen)
    # ICL: self-attn + qassmax, no rope.
    dump_mha("mha_ssmax_self", 64, 8, use_rope=False, ssmax="qassmax-mlp-elementwise",
             q_len=6, kv_len=6, gen=gen)
    # Col ISAB attn1: cross-attn (inducing points -> data) + qassmax, no rope.
    dump_mha("mha_ssmax_cross", 64, 8, use_rope=False, ssmax="qassmax-mlp-elementwise",
             q_len=5, kv_len=9, gen=gen)
    # Col ISAB attn2: cross-attn (data -> inducing points), plain.
    dump_mha("mha_plain_cross", 64, 8, use_rope=False, ssmax="none",
             q_len=9, kv_len=5, gen=gen)
    # Stage 1: ColEmbedding, classification (one-hot encoder, biased LN) and
    # regression (linear encoder, bias-free LN).
    dump_col_embedding(gen, "col_embedding", max_classes=10, bias_free_ln=False)
    dump_col_embedding(gen, "col_embedding_reg", max_classes=0, bias_free_ln=True)
    # Stage 3: ICLearning, classification and regression.
    dump_icl_learning(gen, "icl_learning", max_classes=10, bias_free_ln=False)
    dump_icl_learning(gen, "icl_learning_reg", max_classes=0, bias_free_ln=True)
    # Full model forward, classification and regression.
    dump_full_model(gen, "full_model", max_classes=10, bias_free_ln=False)
    dump_full_model(gen, "full_model_reg", max_classes=0, bias_free_ln=True)
    # Stage 2: full RowInteraction, with and without LayerNorm biases.
    dump_row_interaction(gen, "row_interaction", bias_free_ln=False)
    dump_row_interaction(gen, "row_interaction_biasfree", bias_free_ln=True)
    print(f"\nseed = {SEED}")
    print(f"fixtures in {OUT}")


if __name__ == "__main__":
    main()
