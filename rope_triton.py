import torch

import triton
import triton.language as tl


# int warps_per_block = h < 16 ? 4 : 8;
# dim3 blocks(s, b);
# dim3 threads(THREADS_PER_WARP, warps_per_block);

# TODO; implement with block pointer
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
        BLOCK_SIZE: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    input_block_start = input_ptr + s_id * stride_s + b_id * stride_b
    freq_block_start = freq_ptr + s_id * d2
    output_block_start = output_ptr + s_id * o_stride_s + b_id * o_stride_b
    offsets = tl.arange(0, BLOCK_SIZE)

    # iterate d dimension first not to calculate sin, cos again
    for k in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        freq_shift = k * BLOCK_SIZE
        freq_offsets = freq_shift + offsets
        freq = tl.load(freq_block_start + freq_offsets, mask=freq_offsets < d2)
        cos = tl.cos(freq)
        sin = tl.sin(freq)

        input_d_offsets = offsets * stride_d
        output_d_offsets = offsets * o_stride_d
        for h_id in range(h):
            input_even_block_start = input_block_start + stride_h * h_id
            even = tl.load(input_even_block_start + input_d_offsets, mask=freq_offsets < d2)
            odd = tl.load(input_even_block_start + d2 * stride_d + input_d_offsets, mask=freq_offsets < d2)

            even_result = even * cos - odd * sin
            odd_result = odd * cos + even * sin

            output_even_block_start = output_block_start + o_stride_h * h_id
            tl.store(output_even_block_start + output_d_offsets, even_result, mask=freq_offsets < d2)
            tl.store(output_even_block_start + d2 * o_stride_d + output_d_offsets, odd_result, mask=freq_offsets < d2)

    d_copy_size = d - (2 * d2)
    if 0 < d_copy_size:
        for k in range(0, tl.cdiv(d_copy_size, BLOCK_SIZE)):
            shift = (2 * d2) + (k * BLOCK_SIZE)
            shifted_offsets = shift + offsets
            input_d_offsets = shifted_offsets * stride_d
            output_d_offsets = shifted_offsets * o_stride_d

            for h_id in range(h):
                copy_input_block_start = input_block_start + stride_h * h_id
                input_ = tl.load(copy_input_block_start + input_d_offsets, mask=shifted_offsets < d)
                copy_output_block_start = output_block_start + o_stride_h * h_id
                tl.store(copy_output_block_start + output_d_offsets, input_, mask=shifted_offsets < d)


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

    BLOCK_SIZE = triton.next_power_of_2(d2)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

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
        BLOCK_SIZE=BLOCK_SIZE,
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
        BLOCK_SIZE: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    input_block_start = input_ptr + s_id * stride_s + b_id * stride_b
    freq_block_start = freq_ptr + s_id * d2
    output_block_start = output_ptr + s_id * o_stride_s + b_id * o_stride_b
    offsets = tl.arange(0, BLOCK_SIZE)

    # iterate d dimension first not to calculate sin, cos again
    for k in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        freq_shift = k * BLOCK_SIZE
        freq_offsets = freq_shift + offsets
        mask = freq_offsets < d2
        freq = tl.load(freq_block_start + freq_offsets, mask=mask)
        cos = tl.cos(freq)
        sin = tl.sin(freq)

        input_d_offsets = offsets * stride_d
        output_d_offsets = offsets * o_stride_d
        for h_id in range(h):
            input_even_block_start = input_block_start + stride_h * h_id
            even = tl.load(input_even_block_start + input_d_offsets, mask=mask)
            odd = tl.load(input_even_block_start + d2 * stride_d + input_d_offsets, mask=mask)

            even_result = even * cos + odd * sin
            odd_result = odd * cos - even * sin

            output_even_block_start = output_block_start + o_stride_h * h_id
            tl.store(output_even_block_start + output_d_offsets, even_result, mask=mask)
            tl.store(output_even_block_start + d2 * o_stride_d + output_d_offsets, odd_result, mask=mask)

    d_copy_size = d - (2 * d2)
    if 0 < d_copy_size:
        for k in range(0, tl.cdiv(d_copy_size, BLOCK_SIZE)):
            shift = (2 * d2) + (k * BLOCK_SIZE)
            shifted_offsets = shift + offsets
            mask = shifted_offsets < d
            input_d_offsets = shifted_offsets * stride_d
            output_d_offsets = shifted_offsets * o_stride_d

            for h_id in range(h):
                copy_input_block_start = input_block_start + stride_h * h_id
                input_ = tl.load(copy_input_block_start + input_d_offsets, mask=mask)
                copy_output_block_start = output_block_start + o_stride_h * h_id
                tl.store(copy_output_block_start + output_d_offsets, input_, mask=mask)


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

    BLOCK_SIZE = triton.next_power_of_2(d2)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

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
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
