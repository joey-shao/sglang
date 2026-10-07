"""Aligned per-expert transport slabs preserving the compute tensor's strides."""

from dataclasses import dataclass
from math import lcm

import torch


@dataclass(frozen=True)
class ExpertTensorLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    pitch: int
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, alignment: int):
        if tensor.ndim < 2 or any(n <= 0 for n in tensor.shape):
            raise ValueError("Online EPLB expects nonempty expert-major tensors")
        # Reject overlapping/negative layouts, but preserve transposed and padded
        # UE8M0 scales. Size-one dimensions may carry a large TMA stride.
        span = 1
        for stride, size in sorted(zip(tensor.stride()[1:], tensor.shape[1:])):
            if stride <= 0:
                raise ValueError("Online EPLB requires positive tensor strides")
            if size > 1:
                if stride < span:
                    raise ValueError(
                        "Online EPLB does not support overlapping expert tensors"
                    )
                span += (size - 1) * stride
        if tensor.stride(0) < span:
            raise ValueError("Online EPLB expert storage overlaps")
        elements = lcm(alignment, tensor.element_size()) // tensor.element_size()
        pitch = ((max(span, tensor.stride(0)) + elements - 1) // elements) * elements
        return cls(tuple(tensor.shape[1:]), tensor.stride()[1:], pitch, tensor.dtype)

    def view(self, payload: torch.Tensor) -> torch.Tensor:
        return payload.as_strided(
            (payload.shape[0], *self.shape), (self.pitch, *self.strides)
        )

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        if (tuple(tensor.shape[1:]), tensor.stride()[1:], tensor.dtype) != (
            self.shape,
            self.strides,
            self.dtype,
        ):
            raise ValueError("Online EPLB transport layout changed")
        count = tensor.shape[0]
        available = tensor.untyped_storage().nbytes() // tensor.element_size()
        if (
            tensor.stride(0) == self.pitch
            and tensor.storage_offset() + count * self.pitch <= available
        ):
            return tensor.as_strided((count, self.pitch), (self.pitch, 1))
        # Typically only small scales need padding. Copy their encoded values,
        # never dequantize/requantize or reinterpret a packed scale's meaning.
        payload = torch.zeros(
            (count, self.pitch), dtype=tensor.dtype, device=tensor.device
        )
        self.view(payload).copy_(tensor)
        return payload
