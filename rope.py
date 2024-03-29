from typing import Optional, Tuple, Union

import torch

from rope_triton import (
    rope_forward as forward_v1,
    rope_backward as backward_v1,
)
from rope_triton_v2 import (
    rope_forward as forward_v2,
    rope_backward as backward_v2,
)


class RotaryPositionEmbeddingHalf(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
            self,
            dim: int,
            rotary_percent: float = 1.0,
            seq_len_interpolation_factor: Optional[int] = None,
            pretrained_max_position_embeddings: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        dim: int
            rotary embedding dimension
        rotary_percent: float
            Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor: int
            if not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int
            pre-trained max_position_embeddings before position interpolation
        """
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (
                10000
                ** (
                        torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                        / dim
                )
        )
        self.register_buffer('inv_freq', inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        offset: int, default = 0
            fixed offset for freqencies
        """
        seq = (
                torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
                + offset
        )

        if (self.pretrained_max_position_embeddings is not None
                and self.seq_len_interpolation_factor is not None):
            if (max_seq_len >
                    self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor):
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum('i , j -> i j', seq, self.inv_freq)
        # emb [seq_length, .., dim/2]
        return freqs.reshape(freqs.size(0), 1, 1, freqs.size(1))


class FusedRoPEFuncV1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
    ) -> torch.Tensor:
        if tensor_format == "sbhd":
            output = forward_v1(t, freqs, False)
        elif tensor_format == "bshd":
            output = forward_v1(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = backward_v1(grad_output, freqs, False)
        elif ctx.tensor_format == "bshd":
            grad_input = backward_v1(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None


class FusedRoPEFuncV2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
    ) -> torch.Tensor:
        if tensor_format == "sbhd":
            output = forward_v2(t, freqs, False)
        elif tensor_format == "bshd":
            output = forward_v2(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = backward_v2(grad_output, freqs, False)
        elif ctx.tensor_format == "bshd":
            grad_input = backward_v2(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None
