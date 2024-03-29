import torch

import triton
import triton.language as tl


@triton.jit
def fused_rope_block_forward(
        input_ptr,
        freq_ptr,
        output_ptr,
        h,
        d,
        d2,
        stride_s,
        stride_b,
        stride_h,
        stride_d,
        o_stride_s,
        o_stride_b,
        o_stride_h,
        o_stride_d,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    input_block_start = input_ptr + s_id * stride_s + b_id * stride_b
    freq_block_start = freq_ptr + s_id * d2
    output_block_start = output_ptr + s_id * o_stride_s + b_id * o_stride_b

    offsets_h = tl.arange(0, BLOCK_SIZE_H)
    offsets_d = tl.arange(0, BLOCK_SIZE_D)

    # iterate d dimension first not to calculate sin, cos again
    for k in range(0, tl.cdiv(d2, BLOCK_SIZE_D)):
        freq_shift = k * BLOCK_SIZE_D
        freq_offsets = freq_shift + offsets_d
        freq = tl.load(freq_block_start + freq_offsets, mask=freq_offsets < d2)
        cos = tl.cos(freq)[None, :]
        sin = tl.sin(freq)[None, :]

        for j in range(0, tl.cdiv(h, BLOCK_SIZE_H)):
            head_shift = j * BLOCK_SIZE_H
            head_offsets = head_shift + offsets_h
            mask = (head_offsets < h)[:, None] & (freq_offsets < d2)[None, :]

            input_even_block_start = input_block_start + stride_h * j * BLOCK_SIZE_H
            even_ptr = (offsets_h * stride_h)[:, None] + (offsets_d * stride_d)[None, :]
            even = tl.load(input_even_block_start + even_ptr, mask=mask)
            odd = tl.load(input_even_block_start + d2 * stride_d + even_ptr, mask=mask)

            even_result = even * cos - odd * sin
            odd_result = odd * cos + even * sin

            output_even_block_start = output_block_start + o_stride_h * j * BLOCK_SIZE_H
            o_even_ptr = (offsets_h * o_stride_h)[:, None] + (offsets_d * o_stride_d)[None, :]
            tl.store(output_even_block_start + o_even_ptr, even_result, mask=mask)

            o_odd_ptr = (offsets_h * o_stride_h)[:, None] + (offsets_d * o_stride_d + d2 * o_stride_d)[None, :]
            tl.store(output_even_block_start + o_odd_ptr, odd_result, mask=mask)

    d_copy_size = d - (2 * d2)
    if 0 < d_copy_size:
        for k in range(0, tl.cdiv(d_copy_size, BLOCK_SIZE_D)):
            shift = (2 * d2) + (k * BLOCK_SIZE_D)
            shifted_offsets_d = shift + offsets_d

            for j in range(0, tl.cdiv(h, BLOCK_SIZE_H)):
                head_shift = j * BLOCK_SIZE_H
                head_offsets = head_shift + offsets_h
                mask = (head_offsets < h)[:, None] & (shifted_offsets_d < d)[None, :]

                copy_input_block_start = input_block_start + stride_h * j * BLOCK_SIZE_H
                copy_input_ptr = (offsets_h * stride_h)[:, None] + (shifted_offsets_d * stride_d)[None, :]
                input_ = tl.load(copy_input_block_start + copy_input_ptr, mask=mask)

                copy_output_block_start = output_block_start + o_stride_h * j * BLOCK_SIZE_H
                copy_output_ptr = (offsets_h * o_stride_h)[:, None] + (shifted_offsets_d * o_stride_d)[None, :]
                tl.store(copy_output_block_start + copy_output_ptr, input_, mask=mask)


def rope_forward(t, freq, transpose_output_memory=False):
    s = t.size(0)
    b = t.size(1)
    h = t.size(2)
    d = t.size(3)
    d2 = freq.size(-1)
    stride_s = t.stride(0)
    stride_b = t.stride(1)
    stride_h = t.stride(2)
    stride_d = t.stride(3)

    if transpose_output_memory:
        output = torch.empty((b, s, h, d), device=t.device, dtype=t.dtype).transpose(0, 1)
    else:
        output = torch.empty((s, b, h, d), device=t.device, dtype=t.dtype)

    o_stride_s = output.stride(0)
    o_stride_b = output.stride(1)
    o_stride_h = output.stride(2)
    o_stride_d = output.stride(3)

    BLOCK_SIZE = triton.next_power_of_2(d2 * h)
    if BLOCK_SIZE < 2048:
        num_warps = 4
    elif 2048 <= BLOCK_SIZE < 4096:
        num_warps = 8
    else:
        num_warps = 16

    if d2 < h:
        BLOCK_SIZE_D = triton.next_power_of_2(d2)
        BLOCK_SIZE_H = BLOCK_SIZE // BLOCK_SIZE_D
    else:
        BLOCK_SIZE_H = triton.next_power_of_2(h)
        BLOCK_SIZE_D = BLOCK_SIZE // BLOCK_SIZE_H

    fused_rope_block_forward[(s, b)](
        t,
        freq,
        output,
        h,
        d,
        d2,
        stride_s,
        stride_b,
        stride_h,
        stride_d,
        o_stride_s,
        o_stride_b,
        o_stride_h,
        o_stride_d,
        num_warps=num_warps,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return output


@triton.jit
def fused_rope_block_backward(
        input_ptr,
        freq_ptr,
        output_ptr,
        h,
        d,
        d2,
        stride_s,
        stride_b,
        stride_h,
        stride_d,
        o_stride_s,
        o_stride_b,
        o_stride_h,
        o_stride_d,
        BLOCK_SIZE_D: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    input_block_start = input_ptr + s_id * stride_s + b_id * stride_b
    freq_block_start = freq_ptr + s_id * d2
    output_block_start = output_ptr + s_id * o_stride_s + b_id * o_stride_b

    offsets_h = tl.arange(0, BLOCK_SIZE_H)
    offsets_d = tl.arange(0, BLOCK_SIZE_D)

    # iterate d dimension first not to calculate sin, cos again
    for k in range(0, tl.cdiv(d2, BLOCK_SIZE_D)):
        freq_shift = k * BLOCK_SIZE_D
        freq_offsets = freq_shift + offsets_d
        freq = tl.load(freq_block_start + freq_offsets, mask=freq_offsets < d2)
        cos = tl.cos(freq)[None, :]
        sin = tl.sin(freq)[None, :]

        for j in range(0, tl.cdiv(h, BLOCK_SIZE_H)):
            head_shift = j * BLOCK_SIZE_H
            head_offsets = head_shift + offsets_h
            mask = (head_offsets < h)[:, None] & (freq_offsets < d2)[None, :]

            input_even_block_start = input_block_start + stride_h * j * BLOCK_SIZE_H
            even_ptr = (offsets_h * stride_h)[:, None] + (offsets_d * stride_d)[None, :]
            even = tl.load(input_even_block_start + even_ptr, mask=mask)
            odd = tl.load(input_even_block_start + d2 * stride_d + even_ptr, mask=mask)

            even_result = even * cos + odd * sin
            odd_result = odd * cos - even * sin

            output_even_block_start = output_block_start + o_stride_h * j * BLOCK_SIZE_H
            o_even_ptr = (offsets_h * o_stride_h)[:, None] + (offsets_d * o_stride_d)[None, :]
            tl.store(output_even_block_start + o_even_ptr, even_result, mask=mask)

            o_odd_ptr = (offsets_h * o_stride_h)[:, None] + (offsets_d * o_stride_d + d2 * o_stride_d)[None, :]
            tl.store(output_even_block_start + o_odd_ptr, odd_result, mask=mask)

    d_copy_size = d - (2 * d2)
    if 0 < d_copy_size:
        for k in range(0, tl.cdiv(d_copy_size, BLOCK_SIZE_D)):
            shift = (2 * d2) + (k * BLOCK_SIZE_D)
            shifted_offsets_d = shift + offsets_d

            for j in range(0, tl.cdiv(h, BLOCK_SIZE_H)):
                head_shift = j * BLOCK_SIZE_H
                head_offsets = head_shift + offsets_h
                mask = (head_offsets < h)[:, None] & (shifted_offsets_d < d)[None, :]

                copy_input_block_start = input_block_start + stride_h * j * BLOCK_SIZE_H
                copy_input_ptr = (offsets_h * stride_h)[:, None] + (shifted_offsets_d * stride_d)[None, :]
                input_ = tl.load(copy_input_block_start + copy_input_ptr, mask=mask)

                copy_output_block_start = output_block_start + o_stride_h * j * BLOCK_SIZE_H
                copy_output_ptr = (offsets_h * o_stride_h)[:, None] + (shifted_offsets_d * o_stride_d)[None, :]
                tl.store(copy_output_block_start + copy_output_ptr, input_, mask=mask)


def rope_backward(t, freq, transpose_output_memory=False):
    s = t.size(0)
    b = t.size(1)
    h = t.size(2)
    d = t.size(3)
    d2 = freq.size(-1)
    stride_s = t.stride(0)
    stride_b = t.stride(1)
    stride_h = t.stride(2)
    stride_d = t.stride(3)

    if transpose_output_memory:
        output = torch.empty((b, s, h, d), device=t.device, dtype=t.dtype).transpose(0, 1)
    else:
        output = torch.empty((s, b, h, d), device=t.device, dtype=t.dtype)

    o_stride_s = output.stride(0)
    o_stride_b = output.stride(1)
    o_stride_h = output.stride(2)
    o_stride_d = output.stride(3)

    BLOCK_SIZE = triton.next_power_of_2(d2 * h)
    if BLOCK_SIZE < 2048:
        num_warps = 4
    elif 2048 <= BLOCK_SIZE < 4096:
        num_warps = 8
    else:
        num_warps = 16

    if d2 < h:
        BLOCK_SIZE_D = triton.next_power_of_2(d2)
        BLOCK_SIZE_H = BLOCK_SIZE // BLOCK_SIZE_D
    else:
        BLOCK_SIZE_H = triton.next_power_of_2(h)
        BLOCK_SIZE_D = BLOCK_SIZE // BLOCK_SIZE_H

    fused_rope_block_backward[(s, b)](
        t,
        freq,
        output,
        h,
        d,
        d2,
        stride_s,
        stride_b,
        stride_h,
        stride_d,
        o_stride_s,
        o_stride_b,
        o_stride_h,
        o_stride_d,
        num_warps=num_warps,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return output
