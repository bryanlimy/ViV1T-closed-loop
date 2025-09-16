import math
import typing as t
from typing import Any
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import einsum
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from viv1t.model.core.core import Core
from viv1t.model.core.core import register
from viv1t.model.modules import DropPath


class Image2Patches(nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        patch_size: int,
        stride: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super(Image2Patches, self).__init__()
        assert 1 <= stride <= patch_size
        c, h, w = image_shape
        self.input_shape = image_shape

        num_patches = self.unfold_dim(h, w, patch_size=patch_size, stride=stride)
        patch_dim = patch_size * patch_size * c
        self.projection = nn.Sequential(
            nn.Unfold(kernel_size=patch_size, stride=stride),
            Rearrange("b c l -> b l c"),
            nn.Linear(in_features=patch_dim, out_features=emb_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        num_patches += 1
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, emb_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.num_patches = num_patches
        self.output_shape = (num_patches, emb_dim)

        self.apply(self.init_weight)

    @staticmethod
    def unfold_dim(h: int, w: int, patch_size: int, padding: int = 0, stride: int = 1):
        l = lambda s: math.floor(((s + 2 * padding - patch_size) / stride) + 1)
        return l(h) * l(w)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        patches = self.projection(inputs)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        outputs = torch.cat((cls_tokens, patches), dim=1)
        outputs += self.pos_embedding
        outputs = self.dropout(outputs)
        return outputs


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=use_bias),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        behavior_mode: int,
        out_dim: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super(BehaviorMLP, self).__init__()
        assert behavior_mode in (3, 4)
        in_dim = 2 if behavior_mode == 3 else 4
        self.models = nn.Sequential(
            nn.Linear(
                in_features=in_dim,
                out_features=out_dim // 2,
                bias=use_bias,
            ),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=out_dim // 2,
                out_features=out_dim,
                bias=use_bias,
            ),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor):
        return self.models(inputs)


class Attention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_bias: bool = True,
        flash_attention: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Attention, self).__init__()
        self.flash_attention = flash_attention
        self.grad_checkpointing = grad_checkpointing
        self.dropout = dropout
        inner_dim = emb_dim * num_heads

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.to_qkv = nn.Linear(
            in_features=emb_dim, out_features=inner_dim * 3, bias=False
        )
        self.rearrange = Rearrange("b n (h d) -> b h n d", h=num_heads)

        self.projection = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim, bias=use_bias),
            nn.Dropout(p=dropout),
        )

        self.register_buffer("scale", torch.tensor(emb_dim**-0.5))

        if not self.flash_attention:
            self.softmax = nn.Softmax(dim=-1)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if self.flash_attention:
            # Adding dropout to Flash attention layer significantly increase memory usage
            outputs = F.scaled_dot_product_attention(q, k, v)
        else:
            attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn = self.softmax(attn_weights)
            outputs = torch.matmul(attn, v)
        outputs = F.dropout(outputs, p=self.dropout, training=self.training)
        return outputs

    def mha(self, inputs: torch.Tensor):
        inputs = self.layer_norm(inputs)
        q, k, v = torch.chunk(self.to_qkv(inputs), chunks=3, dim=-1)
        outputs = self.scaled_dot_product_attention(
            q=self.rearrange(q), k=self.rearrange(k), v=self.rearrange(v)
        )
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.projection(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.mha, inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = self.mha(inputs)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        emb_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        behavior_mode: int,
        drop_path: float = 0.0,
        use_bias: bool = True,
        flash_attention: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "mha": Attention(
                        emb_dim=emb_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_bias=use_bias,
                        flash_attention=flash_attention,
                        grad_checkpointing=grad_checkpointing,
                    ),
                    "mlp": MLP(
                        in_dim=emb_dim,
                        hidden_dim=mlp_dim,
                        dropout=dropout,
                        use_bias=use_bias,
                    ),
                }
            )
            if behavior_mode in (3, 4):
                block["b-mlp"] = BehaviorMLP(
                    behavior_mode=behavior_mode,
                    out_dim=emb_dim,
                    use_bias=use_bias,
                )
            self.blocks.append(block)
        self.drop_path = DropPath(p=drop_path)
        self.output_shape = (input_shape[0], emb_dim)

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs: torch.Tensor, behaviors: torch.Tensor):
        outputs = inputs
        for block in self.blocks:
            if "b-mlp" in block:
                b_latent = block["b-mlp"](behaviors)
                b_latent = repeat(b_latent, "b d -> b 1 d")
                outputs = outputs + b_latent
            outputs = self.drop_path(block["mha"](outputs)) + outputs
            outputs = self.drop_path(block["mlp"](outputs)) + outputs
        return outputs


class V1T(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, int, int]):
        super(V1T, self).__init__()
        self.register_buffer("reg_scale", torch.tensor(0.0))
        self.behavior_mode = args.core_behavior_mode

        if args.grad_checkpointing is None:
            args.grad_checkpointing = "cuda" in args.device.type
        if args.grad_checkpointing and args.verbose:
            print(f"Enable gradient checkpointing in V1T")

        self.patch_embedding = Image2Patches(
            image_shape=input_shape,
            patch_size=args.core_patch_size,
            stride=args.core_patch_stride,
            emb_dim=args.core_emb_dim,
            dropout=args.core_p_dropout,
        )
        self.transformer = Transformer(
            input_shape=self.patch_embedding.output_shape,
            emb_dim=args.core_emb_dim,
            num_blocks=args.core_num_blocks,
            num_heads=args.core_num_heads,
            mlp_dim=args.core_mlp_dim,
            dropout=args.core_t_dropout,
            behavior_mode=args.core_behavior_mode,
            drop_path=args.core_drop_path,
            use_bias=not args.core_disable_bias,
            flash_attention=bool(args.core_flash_attention),
            grad_checkpointing=args.grad_checkpointing,
        )
        # calculate latent height and width based on num_patches
        h, w = self.find_shape(self.patch_embedding.num_patches - 1)
        self.rearrange = Rearrange("b (h w) c -> b c h w", h=h, w=w)
        self.output_shape = (self.transformer.output_shape[-1], h, w)

    @staticmethod
    def find_shape(num_patches: int):
        dim1 = math.ceil(math.sqrt(num_patches))
        while num_patches % dim1 != 0 and dim1 > 0:
            dim1 -= 1
        dim2 = num_patches // dim1
        return dim1, dim2

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = self.patch_embedding(inputs)
        if self.behavior_mode == 4:
            behaviors = torch.cat((behaviors, pupil_centers), dim=-1)
        outputs = self.transformer(outputs, behaviors=behaviors)
        outputs = outputs[:, 1:, :]  # remove CLS token
        outputs = self.rearrange(outputs)
        return outputs


@register("v1t")
class V1TCore(Core):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        verbose: int = 0,
    ):
        super(V1TCore, self).__init__(args, input_shape=input_shape, verbose=verbose)
        self.input_shape = input_shape
        self.behavior_mode = args.core_behavior_mode
        input_shape = list(input_shape)
        t = input_shape.pop(1)  # remove time dimension
        match self.behavior_mode:
            case 0 | 3 | 4:
                pass
            case 1:
                input_shape[0] += 2
            case 2:
                input_shape[0] += 4
            case _:
                raise NotImplementedError(f"--behavior_mode {self.behavior_mode}")
        self.v1t = V1T(args, input_shape=tuple(input_shape))
        self.output_shape = list(self.v1t.output_shape)
        self.output_shape.insert(1, t)

    def regularizer(self):
        return self.v1t.regularizer()

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, _, t, h, w = inputs.shape
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
        outputs = rearrange(outputs, "b c t h w -> (b t) c h w")
        behaviors = rearrange(behaviors, "b d t -> (b t) d")
        pupil_centers = rearrange(pupil_centers, "b d t -> (b t) d")
        outputs = self.v1t(
            inputs=outputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = rearrange(outputs, "(b t) c h w -> b c t h w", b=b)
        return outputs
