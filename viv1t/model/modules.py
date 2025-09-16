import math
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.ops import stochastic_depth


class ELU1(nn.Module):
    """ELU + 1 activation to output standardized responses"""

    def __init__(self):
        super(ELU1, self).__init__()
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor):
        return F.elu(inputs) + self.one


class Exponential(nn.Module):
    """Exponential activation to output standardized responses"""

    def __init__(self):
        super(Exponential, self).__init__()

    def forward(self, inputs: torch.Tensor):
        return torch.exp(inputs)


class LearnableSoftplus(nn.Module):
    """Softplus activation with a learnable beta parameter

    Reference: https://github.com/lRomul/sensorium?tab=readme-ov-file#learnable-softplus
    """

    def __init__(self, beta: float = 1.0, threshold: int = 20):
        super(LearnableSoftplus, self).__init__()
        self.register_parameter(
            name="log_beta", param=torch.nn.Parameter(torch.log(torch.tensor(beta)))
        )
        self.register_buffer(name="threshold", tensor=torch.tensor(threshold))

    def forward(self, inputs: torch.Tensor):
        beta = torch.exp(self.log_beta)
        beta_x = beta * inputs
        return torch.where(
            beta_x < self.threshold, torch.log1p(torch.exp(beta_x)) / beta, inputs
        )


class OutputActivation(nn.ModuleDict):
    def __init__(
        self,
        output_mode: int,
        neuron_coordinates: dict[str, np.ndarray | torch.Tensor],
    ):
        super(OutputActivation, self).__init__()
        match output_mode:
            case 0:
                output_activation = nn.Identity
            case 1:
                output_activation = ELU1
            case 2:
                output_activation = Exponential
            case 3:
                output_activation = LearnableSoftplus
            case 4:
                output_activation = nn.Sigmoid
            case _:
                raise NotImplementedError(f"output_mode {output_mode} not implemented.")
        for mouse_id in neuron_coordinates.keys():
            self.add_module(name=mouse_id, module=output_activation())

    def forward(self, inputs: torch.Tensor, mouse_id: str):
        return self[mouse_id](inputs)


class SwiGLU(nn.Module):
    """
    SwiGLU activation by Shazeer et al. 2022
    https://arxiv.org/abs/2002.05202
    """

    def forward(self, inputs: torch.Tensor):
        outputs, gate = inputs.chunk(2, dim=-1)
        return F.silu(gate) * outputs


class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift: int, yshift: int, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)
        self.x_shift = torch.tensor(xshift, dtype=torch.float)
        self.y_shift = torch.tensor(yshift, dtype=torch.float)
        self.elu = nn.ELU()

    def forward(self, inputs: torch.Tensor):
        return self.elu(inputs - self.x_shift) + self.y_shift


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate"""

    def __init__(
        self,
        size: Union[int, Tuple[int, int], Tuple[int, int, int]],
        mode: str = "nearest",
        antialias: bool = False,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.antialias = antialias

    def forward(self, inputs: torch.Tensor):
        return F.interpolate(
            inputs,
            size=self.size,
            mode=self.mode,
            antialias=self.antialias,
        )


class DropPath(nn.Module):
    """Stochastic depth for regularization https://arxiv.org/abs/1603.09382"""

    def __init__(self, p: float = 0.0, mode: str = "row"):
        super(DropPath, self).__init__()
        assert 0 <= p <= 1
        assert mode in ("batch", "row")
        self.p, self.mode = p, mode

    def forward(self, inputs: torch.Tensor):
        return stochastic_depth(
            inputs, p=self.p, mode=self.mode, training=self.training
        )


def get_activation(name: str) -> nn.Module:
    """Get activation function"""
    match name:
        case "none":
            activation = nn.Identity
        case "tanh":
            activation = nn.Tanh
        case "elu":
            activation = nn.ELU
        case "relu":
            activation = nn.ReLU
        case "gelu":
            activation = nn.GELU
        case "silu":
            activation = nn.SiLU
        case "swiglu":
            activation = SwiGLU
        case _:
            raise NotImplementedError(f"Activation function {name} not implemented.")
    return activation


def get_ff_activation(name: str, ff_dim: int) -> (nn.Module, int):
    """Get activation function and dimension for Transformer FeedForward module"""
    match name:
        case "none":
            ff_out = ff_dim
            activation = nn.Identity
        case "tanh":
            ff_out = ff_dim
            activation = nn.Tanh
        case "elu":
            ff_out = ff_dim
            activation = nn.ELU
        case "relu":
            ff_out = ff_dim
            activation = nn.ReLU
        case "gelu":
            ff_out = ff_dim
            activation = nn.GELU
        case "silu":
            ff_out = ff_dim
            activation = nn.SiLU
        case "swiglu":
            ff_out = ff_dim * 2
            activation = SwiGLU
        case _:
            raise NotImplementedError(f"FF {name} not implemented.")
    return activation, ff_out


class ShardLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ShardLinear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.init_weight()

    def init_weight(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor):
        b, t, p, c = inputs.shape
        step = (b * t * p) // 12
        inputs = rearrange(inputs, "b t p c -> (b t p) c")
        outputs = torch.zeros((inputs.size(0), self.out_features), device=inputs.device)
        for i in range(0, outputs.size(0), step):
            outputs[i : i + step] = inputs[i : i + step] @ self.weight.t() + self.bias
        outputs = rearrange(outputs, "(b t p) c -> b t p c", b=b, t=t, p=p)
        return outputs


class Unfold3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        norm: str = "layernorm",
    ):
        super(Unfold3d, self).__init__()
        assert len(kernel_size) == len(stride) == 3
        self.kernel_size, self.stride = kernel_size, stride

        patch_dim = int(in_channels * np.prod(kernel_size))
        self.rearrange = Rearrange("b c nt nh nw pt ph pw -> b nt (nh nw) (c pt ph pw)")
        self.norm = get_norm(norm)(patch_dim)
        self.linear = nn.Linear(in_features=patch_dim, out_features=out_channels)

    def forward(self, inputs: torch.Tensor):
        outputs = (
            inputs.unfold(2, size=self.kernel_size[0], step=self.stride[0])
            .unfold(3, size=self.kernel_size[1], step=self.stride[1])
            .unfold(4, size=self.kernel_size[2], step=self.stride[2])
        )
        outputs = self.rearrange(outputs)
        outputs = self.norm(outputs)
        outputs = self.linear(outputs)
        return outputs


class UnfoldConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        norm: str = "layernorm",
        dilation: int = 1,
        padding: int = 0,
    ):
        super(UnfoldConv3d, self).__init__()
        assert len(kernel_size) == len(stride) == 3
        self.in_channels = in_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.dilation, self.padding = dilation, padding

        # prepare one-hot convolution kernel
        kernel_dim = int(np.prod(self.kernel_size))
        repeat = [in_channels, 1] + [1] * len(self.kernel_size)
        self.register_buffer(
            "weight",
            torch.eye(kernel_dim)
            .reshape((kernel_dim, 1, *self.kernel_size))
            .repeat(*repeat),
        )

        patch_dim = int(in_channels * np.prod(kernel_size))
        self.rearrange = Rearrange("b c t h w -> b t (h w) c")
        self.norm = get_norm(norm)(patch_dim)
        self.linear = nn.Linear(in_features=patch_dim, out_features=out_channels)

    def forward(self, inputs: torch.Tensor):
        outputs = F.conv3d(
            inputs,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        outputs = self.rearrange(outputs)
        outputs = self.norm(outputs)
        outputs = self.linear(outputs)
        return outputs


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        reg_tokens: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        ff_dropout: float,
        drop_path: float,
        use_rope: bool,
        flash_attention: bool,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
        grad_checkpointing: bool,
    ):
        super(TransformerBlock, self).__init__()
        self.input_shape = input_shape
        self.reg_tokens = reg_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.ff_activation = ff_activation
        self.mha_dropout = mha_dropout
        self.ff_dropout = ff_dropout
        self.drop_path = drop_path
        self.use_rope = use_rope
        self.flash_attention = flash_attention
        self.is_causal = is_causal
        self.normalize_qk = normalize_qk
        self.grad_checkpointing = grad_checkpointing

        self.emb_dim = input_shape[-1]
        self.inner_dim = head_dim * num_heads

        self.register_buffer("scale", torch.tensor(head_dim**-0.5))

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if self.flash_attention:
            # Adding dropout to Flash attention layer significantly increase memory usage
            outputs = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        else:
            l, s = q.size(-2), k.size(-2)
            attn_bias = torch.zeros(l, s, dtype=q.dtype, device=q.device)
            if self.is_causal:
                mask = torch.ones(l, s, dtype=torch.bool, device=q.device).tril(0)
                attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                attn_bias.to(q.dtype)
            attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn_weights += attn_bias
            attn = torch.softmax(attn_weights, dim=-1)
            outputs = torch.matmul(attn, v)
        outputs = F.dropout(outputs, p=self.mha_dropout, training=self.training)
        return outputs


class RotaryPosEmb(nn.Module):
    """
    Rotary position embedding (RoPE)
    Reference
    - Su et al. 2021 https://arxiv.org/abs/2104.09864
    - Sun et al. 2022 https://arxiv.org/abs/2212.10554
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        reg_tokens: int,
        scale_base: int = 512,
        use_xpos: bool = True,
    ):
        super(RotaryPosEmb, self).__init__()
        self.num_tokens = num_tokens
        self.reg_tokens = reg_tokens

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        )
        self.create_embedding(n=num_tokens)

    def create_embedding(self, n: int):
        device = self.scale.device
        t = torch.arange(n, dtype=self.inv_freq.dtype, device=device)
        freq = torch.einsum("i , j -> i j", t, self.inv_freq)
        freq = torch.cat((freq, freq), dim=-1)
        if self.use_xpos:
            power = (t - (n // 2)) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)
        else:
            scale = torch.ones(1, device=device)
        self.register_buffer("emb_sin", torch.sin(freq), persistent=False)
        self.register_buffer("emb_cos", torch.cos(freq), persistent=False)
        self.register_buffer("emb_scale", scale, persistent=False)

    def get_embedding(self, n: int):
        if self.emb_sin is None or self.emb_sin.shape[-2] < n:
            self.create_embedding(n)
        return self.emb_sin[:n], self.emb_cos[:n], self.emb_scale[:n]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def rotate(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            (q * cos * scale) + (cls.rotate_half(q) * sin * scale),
            (k * cos * scale) + (cls.rotate_half(k) * sin * scale),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        n, device = q.size(2), q.device.type
        q_reg, k_reg = None, None
        if self.reg_tokens:
            q_reg = q[:, :, -self.reg_tokens :, :]
            k_reg = k[:, :, -self.reg_tokens :, :]
            n -= self.reg_tokens
            q = q[:, :, : -self.reg_tokens, :]
            k = k[:, :, : -self.reg_tokens, :]
        sin, cos, scale = self.get_embedding(n)
        q, k = self.rotate(q, k, sin, cos, scale)
        if q_reg is not None and k_reg is not None:
            q = torch.cat((q, q_reg), dim=2)
            k = torch.cat((k, k_reg), dim=2)
        return q, k


class SinCosPosEmb(nn.Module):
    def __init__(self, emb_dim: int, input_shape: Tuple[int, int]):
        super(SinCosPosEmb, self).__init__()
        assert emb_dim % 2 == 0, f"emb_dim must be divisible by 2, got {emb_dim}."
        self.emb_dim = emb_dim

        h, w = input_shape
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])
        emb_h = self._1d_sin_cos_pos_emb(self.emb_dim // 2, pos=grid[0])
        emb_w = self._1d_sin_cos_pos_emb(self.emb_dim // 2, pos=grid[1])
        pos_emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

        self.register_buffer("pos_emb", pos_emb, persistent=False)

    @staticmethod
    def _1d_sin_cos_pos_emb(emb_dim: int, pos: torch.Tensor):
        omega = torch.arange(emb_dim // 2, dtype=torch.float32)
        omega /= emb_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = torch.flatten(pos)
        out = einsum(pos, omega, "m, d -> m d")
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb

    def forward(self, inputs: torch.Tensor):
        b, t, p, d = inputs.shape
        return inputs + self.pos_emb[None, None, :p]


class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int,
        dimension: Literal["spatial", "temporal"],
        dropout: float = 0.0,
    ):
        super(SinusoidalPosEmb, self).__init__()
        # input has shape (B, T, P, D)
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        match self.dimension:
            case "temporal":
                pos_encoding = torch.zeros(1, max_length, 1, d_model)
                pos_encoding[0, :, 0, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, :, 0, 1::2] = torch.cos(position * div_term)
            case "spatial":
                pos_encoding = torch.zeros(1, 1, max_length, d_model)
                pos_encoding[0, 0, :, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, 0, :, 1::2] = torch.cos(position * div_term)
            case _:
                raise NotImplementedError(
                    f"invalid dimension {self.dimension} in "
                    f"SinusoidalPositionalEncoding"
                )

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        match self.dimension:
            case "temporal":
                outputs += self.pos_encoding[:, : inputs.size(1), :, :]
            case "spatial":
                outputs += self.pos_encoding[:, :, : inputs.size(2), :]
        return self.dropout(outputs)


class ConcatBehaviors(nn.Module):
    """
    Concatenate behavior and pupil center features with the visual input in the
    channel dimension

    behavior_mode:
        0: No concatenation
        1: Concatenate behavior features
        2: Concatenate behavior and pupil center features
    """

    def __init__(self, input_shape: Tuple[int, ...], behavior_mode: int):
        super(ConcatBehaviors, self).__init__()
        assert len(input_shape) == 4, f"input_shape must be in format (C, T, H W)"
        assert behavior_mode in (0, 1, 2)
        self.input_shape = input_shape
        self.behavior_mode = behavior_mode
        d, t, h, w = input_shape
        match self.behavior_mode:
            case 1:
                d += 2
            case 2:
                d += 4
        self.output_shape = (d, t, h, w)

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        _, _, _, h, w = inputs.shape
        outputs = inputs
        match self.behavior_mode:
            case 1:
                outputs = torch.cat(
                    (
                        outputs,
                        repeat(behaviors, "b d t -> b d t h w", h=h, w=w),
                    ),
                    dim=1,
                )
            case 2:
                outputs = torch.cat(
                    (
                        outputs,
                        repeat(behaviors, "b d t -> b d t h w", h=h, w=w),
                        repeat(pupil_centers, "b d t -> b d t h w", h=h, w=w),
                    ),
                    dim=1,
                )
        return outputs


class DropPatch(nn.Module):
    """
    Apply dropout to spatial and temporal patches where the entire patch
    is zeroed out
    """

    def __init__(self, p: float = 0.0):
        super(DropPatch, self).__init__()
        self.p = p

    def forward(self, inputs: torch.Tensor):
        outputs, t = inputs, inputs.size(1)
        outputs = rearrange(outputs, "b t p c -> b (t p) c")
        outputs = F.dropout1d(outputs, p=self.p, training=self.training)
        outputs = rearrange(outputs, "b (t p) c -> b t p c", t=t)
        return outputs


class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) from Zhu et al. 2025
    Reference:
    - https://arxiv.org/abs/2503.10622
    - https://jiachenzhu.github.io/DyT/
    """

    def __init__(self, num_features: int, alpha_init_value: float = 0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.tanh(self.alpha * inputs)
        return outputs * self.weight + self.bias


def get_norm(name: str):
    match name.lower():
        case "layernorm":
            return nn.LayerNorm
        case "rmsnorm":
            return nn.RMSNorm
        case "dyt":
            return DyT
        case _:
            raise NotImplementedError(f"Norm {name} not implemented.")
