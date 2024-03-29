import os

import torch
import triton

from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)

from rope import FusedRoPEFuncV2, RotaryPositionEmbeddingHalf


def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)


def te_fused_forward_backward(t, seq_length, rotary_pos_emb, loss_func, tensor_format):
    emb = rotary_pos_emb(seq_length)
    output_te = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=True,
    )
    loss_te = loss_func(output_te)
    loss_te.backward()


def torch_native_forward_backward(t, seq_length, rotary_pos_emb, loss_func, tensor_format):
    emb = rotary_pos_emb(seq_length)
    output_te = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=False,
    )
    loss_te = loss_func(output_te)
    loss_te.backward()


def triton_forward_backward(t, seq_length, rotary_pos_emb, loss_func, tensor_format, version=1):
    emb = rotary_pos_emb(seq_length)
    output_tri = FusedRoPEFuncV2.apply(t, emb, tensor_format)
    loss_tri = loss_func(output_tri)
    loss_tri.backward()


defaults = {
    "seq_length": 4096,
    "hidden_size": 128,
    "rotary_percent": 1.0,
    "batch_size": 16,
    "head_num": 32,
    "margin": 0
}
x_vals = {
    "seq_length": [512, 1024, 2048, 4096],
    "hidden_size": [128, 256, 512],
    "rotary_percent": [0.5, 1.0],
    "batch_size": [2, 4, 8, 16],
    "head_num": [8, 16, 32, 64],
    "margin": [0, 10, 33, 77],
}
configs = []
for key in defaults:
    args = {k: v for k, v in defaults.items() if k != key}
    configs.append(
        triton.testing.Benchmark(
            x_names=[key],
            x_vals=x_vals[key],
            line_arg="provider",
            line_vals=[
                'torch',
                'te',
                'triton',
            ],  # possible values for `line_arg``
            line_names=[
                "Torch Native",
                "Transformer Engine (Fused)",
                "Triton",
            ],  # label name for the lines
            styles=[('red', '-'), ('blue', '-'), ('green', '-')],  # line styles
            ylabel="runtime",  # label name for the y-axis
            plot_name=f"RoPE-performance-version-2-{key}-test",
            args=args,
        )
    )


@triton.testing.perf_report(configs)
def benchmark(
    batch_size,
    head_num,
    seq_length,
    hidden_size,
    rotary_percent,
    margin,
    provider,
    transpose=None,
    tensor_format="sbhd",
    loss_func=_non_overlapping_grad,
):
    dtype = torch.float16
    device = torch.device("cuda:0")
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    warmup = 25
    rep = 500
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_native_forward_backward(t, seq_length, rotary_pos_emb, loss_func, tensor_format),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
            grad_to_none=[t],
        )

    elif provider == 'te':
        rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: te_fused_forward_backward(t, seq_length, rotary_pos_emb, loss_func, tensor_format),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
            grad_to_none=[t],
        )

    elif provider == 'triton':
        rotary_pos_emb = RotaryPositionEmbeddingHalf(hidden_size, rotary_percent)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_forward_backward(t, seq_length, rotary_pos_emb, loss_func, tensor_format, version=1),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
            grad_to_none=[t],
        )

    else:
        raise NotImplementedError

    return ms, max_ms, min_ms


result_path = "./version-2"
os.makedirs(result_path, exist_ok=True)
benchmark.run(save_path=result_path, print_data=True)
