import math
from typing import Any
from typing import Tuple

import torch
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from viv1t.model.core.core import Core
from viv1t.model.core.core import register
from viv1t.model.modules import ConcatBehaviors
from viv1t.model.modules import DropPatch
from viv1t.model.modules import DropPath
from viv1t.model.modules import RotaryPosEmb
from viv1t.model.modules import SinCosPosEmb
from viv1t.model.modules import SinusoidalPosEmb
from viv1t.model.modules import TransformerBlock
from viv1t.model.modules import Unfold3d
from viv1t.model.modules import UnfoldConv3d
from viv1t.model.modules import get_activation
from viv1t.model.modules import get_ff_activation
from viv1t.model.modules import get_norm


class Tokenizer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        behavior_mode: int,
        patch_mode: int,
        spatial_patch_size: int,
        spatial_patch_stride: int,
        temporal_patch_size: int,
        temporal_patch_stride: int,
        pos_encoding: int,
        emb_dim: int,
        dropout: float,
        max_frame: int,
        pad_frame: bool,
        norm: str,
    ):
        """
        Patch mode (--core_patch_mode)
            0: extract 3D patches via tensor.unfold followed by linear projection
            1: extract 3D patches via a 3D convolution layer
        """
        super(Tokenizer, self).__init__()
        self.input_shape = input_shape
        self.behavior_mode = behavior_mode
        c, t, h, w = input_shape
        t = t if max_frame is None else max_frame

        self.concat_behaviors = ConcatBehaviors(input_shape, self.behavior_mode)
        c = self.concat_behaviors.output_shape[0]

        self.pad, self.padding = None, (0, 0, 0, 0, 0, 0)
        h_pad = self.pad_size(h, spatial_patch_size, stride=spatial_patch_stride)
        w_pad = self.pad_size(w, spatial_patch_size, stride=spatial_patch_stride)
        t_pad = self.pad_size(t, temporal_patch_size, stride=temporal_patch_stride)
        if pad_frame:
            # zero padding in the spatial dimensions to ensure that the entire input is covered
            self.padding = (
                w_pad // 2,  # padding left
                w_pad - w_pad // 2,  # padding right
                h_pad // 2,  # padding top
                h_pad - h_pad // 2,  # padding bottom
                t_pad,  # padding front
                0,  # padding back
            )
            self.pad = nn.ZeroPad3d(self.padding)
            # self.pad = nn.ReplicationPad3d(self.padding)
            # dimension of input after zero padding
            w = w + self.padding[0] + self.padding[1]
            h = h + self.padding[2] + self.padding[3]
            t = t + self.padding[4] + self.padding[5]
        elif h_pad > 0 or w_pad > 0:
            print(
                "Warning: the patches do not cover the entire input frame, "
                "set --core_pad_frame=1 to zero pad the frame."
            )

        new_t = self.unfold_size(t, temporal_patch_size, stride=temporal_patch_stride)
        new_h = self.unfold_size(h, spatial_patch_size, stride=spatial_patch_stride)
        new_w = self.unfold_size(w, spatial_patch_size, stride=spatial_patch_stride)

        self.kernel_size = (temporal_patch_size, spatial_patch_size, spatial_patch_size)
        self.stride = (
            temporal_patch_stride,
            spatial_patch_stride,
            spatial_patch_stride,
        )

        match patch_mode:
            case 0:
                self.tokenizer = Unfold3d(
                    in_channels=c,
                    out_channels=emb_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    norm=norm,
                )
            case 1:
                self.tokenizer = UnfoldConv3d(
                    in_channels=c,
                    out_channels=emb_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    norm=norm,
                )
            case 2:
                self.tokenizer = nn.Sequential(
                    nn.Conv3d(
                        in_channels=c,
                        out_channels=emb_dim,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                    ),
                    Rearrange("b c t h w -> b t (h w) c"),
                )
            case _:
                raise NotImplementedError(
                    f"--core_patch_mode {patch_mode} not implemented."
                )

        self.norm = get_norm(norm)(emb_dim)
        self.drop_patch = DropPatch(p=dropout)

        self.pos_encoding = pos_encoding
        match self.pos_encoding:
            case 1:
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, new_t, new_h * new_w, emb_dim)
                )
            case 2:
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, 1, new_h * new_w, emb_dim)
                )
                self.temporal_pos_embedding = nn.Parameter(
                    torch.randn(1, new_t, 1, emb_dim)
                )
            case 3:
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, 1, new_h * new_w, emb_dim)
                )
                self.temporal_pos_encoding = SinusoidalPosEmb(
                    d_model=emb_dim,
                    max_length=max_frame,
                    dimension="temporal",
                    dropout=dropout,
                )
            case 4:
                self.spatial_pos_embedding = SinusoidalPosEmb(
                    d_model=emb_dim,
                    max_length=new_h * new_w,
                    dimension="spatial",
                    dropout=dropout,
                )
                self.temporal_pos_encoding = SinusoidalPosEmb(
                    d_model=emb_dim,
                    max_length=max_frame,
                    dimension="temporal",
                    dropout=dropout,
                )
            case 6:
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, 1, new_h * new_w, emb_dim)
                )
            case 7:
                self.spatial_pos_embedding = SinCosPosEmb(
                    emb_dim=emb_dim, input_shape=(new_h, new_w)
                )
        self.new_shape = (new_h, new_w)
        self.output_shape = (new_t, (new_h * new_w), emb_dim)

    @staticmethod
    def pad_size(dim: int, patch_size: int, stride: int = 1):
        """Compute the zero padding needed to cover the entire dimension"""
        return (math.ceil(dim / stride) - 1) * stride + patch_size - dim

    @staticmethod
    def unfold_size(dim: int, patch_size: int, stride: int = 1):
        return math.floor(((dim - patch_size) / stride) + 1)

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        outputs = self.concat_behaviors(outputs, behaviors, pupil_centers)
        if self.pad is not None:
            outputs = self.pad(outputs)
        outputs = self.tokenizer(outputs)
        outputs = self.norm(outputs)
        outputs = self.drop_patch(outputs)
        _, t, p, _ = outputs.shape
        match self.pos_encoding:
            case 1:
                outputs = outputs + self.pos_embedding[:, :t, :p, :]
            case 2:
                # TODO check maximum temporal dimension in the bonus test set
                outputs = (
                    outputs
                    + self.spatial_pos_embedding[:, :, :p, :]
                    + self.temporal_pos_embedding[:, :t, :, :]
                )
            case 3:
                outputs = outputs + self.spatial_pos_embedding[:, :, :p, :]
                outputs = self.temporal_pos_encoding(outputs)
            case 4:
                outputs = self.spatial_pos_embedding(outputs)
                outputs = self.temporal_pos_encoding(outputs)
            case 6:
                outputs += self.spatial_pos_embedding[:, :, :p, :]
            case 7:
                outputs = self.spatial_pos_embedding(outputs)
        return outputs


class ParallelAttentionBlock(TransformerBlock):
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
        super(ParallelAttentionBlock, self).__init__(
            input_shape=input_shape,
            reg_tokens=reg_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            drop_path=drop_path,
            use_rope=use_rope,
            flash_attention=flash_attention,
            is_causal=is_causal,
            norm=norm,
            normalize_qk=normalize_qk,
            grad_checkpointing=grad_checkpointing,
        )
        self.norm = get_norm(norm)(self.emb_dim)
        ff_activation, ff_out = get_ff_activation(ff_activation, ff_dim)
        self.fused_dims = (self.inner_dim, self.inner_dim, self.inner_dim, ff_out)
        self.fused_linear = nn.Linear(
            in_features=self.emb_dim, out_features=sum(self.fused_dims), bias=False
        )
        self.attn_out = nn.Linear(
            in_features=self.inner_dim, out_features=self.emb_dim, bias=False
        )
        self.ff_out = nn.Sequential(
            ff_activation(),
            nn.Dropout(p=ff_dropout),
            nn.Linear(in_features=ff_dim, out_features=self.emb_dim, bias=False),
        )
        self.drop_path1 = DropPath(p=drop_path)
        self.drop_path2 = DropPath(p=drop_path)

        self.normalize_qk = normalize_qk
        if self.normalize_qk:
            self.norm_q = get_norm(norm)(self.inner_dim)
            self.norm_k = get_norm(norm)(self.inner_dim)

        if self.use_rope:
            self.rotary_position_embedding = RotaryPosEmb(
                dim=head_dim, num_tokens=input_shape[0], reg_tokens=reg_tokens
            )

        self.apply(self.init_weight)

    def parallel_attention(self, inputs: torch.Tensor):
        outputs = self.norm(inputs)
        q, k, v, ff = self.fused_linear(outputs).split(self.fused_dims, dim=-1)
        if self.normalize_qk:
            q, k = self.norm_q(q), self.norm_k(k)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.use_rope:
            q, k = self.rotary_position_embedding(q=q, k=k)

        attn = self.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b h n d -> b n (h d)")
        outputs = (
            inputs
            + self.drop_path1(self.attn_out(attn))
            + self.drop_path2(self.ff_out(ff))
        )
        return outputs

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        outputs = inputs
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.parallel_attention,
                outputs,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        else:
            outputs = self.parallel_attention(outputs)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        reg_tokens: int,
        depth: int,
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
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ParallelAttentionBlock(
                    input_shape=input_shape,
                    reg_tokens=reg_tokens,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    ff_activation=ff_activation,
                    mha_dropout=mha_dropout,
                    ff_dropout=ff_dropout,
                    drop_path=drop_path,
                    use_rope=use_rope,
                    flash_attention=flash_attention,
                    is_causal=is_causal,
                    norm=norm,
                    normalize_qk=normalize_qk,
                    grad_checkpointing=grad_checkpointing,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        for i in range(len(self.blocks)):
            outputs = self.blocks[i](
                outputs,
                mouse_id=mouse_id,
                behaviors=behaviors,
                pupil_centers=pupil_centers,
            )
        return outputs


class ViViT(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, ...], verbose: int = 0):
        super(ViViT, self).__init__()
        self.register_buffer("reg_scale", torch.tensor(0.0))
        self.behavior_mode = args.core_behavior_mode
        pos_encoding = args.core_pos_encoding
        emb_dim, num_heads = args.core_emb_dim, args.core_num_heads

        if args.grad_checkpointing is None:
            args.grad_checkpointing = "cuda" in args.device.type
        if args.grad_checkpointing and verbose:
            print(f"Enable gradient checkpointing in ViViT")

        args.core_parallel_attention = True
        if args.core_parallel_attention and verbose:
            print(f"Use parallel attention and MLP in ViViT.")

        if not hasattr(args, "core_reg_tokens"):
            args.core_reg_tokens = 0
        self.reg_tokens = args.core_reg_tokens
        if self.reg_tokens > 0:
            self.reg_s_tokens = nn.Parameter(torch.randn(self.reg_tokens, emb_dim))
            self.reg_t_tokens = nn.Parameter(torch.randn(self.reg_tokens, emb_dim))
        else:
            self.reg_s_tokens = None
            self.reg_t_tokens = None

        if not hasattr(args, "core_use_causal_attention"):
            args.core_use_causal_attention = False
        if args.core_use_causal_attention and verbose:
            print(f"Enable causal attention in temporal Transformer.")

        normalize_qk = hasattr(args, "core_norm_qk") and args.core_norm_qk

        self.spatial_transformer = Transformer(
            input_shape=(input_shape[1], emb_dim),
            reg_tokens=self.reg_tokens,
            depth=args.core_spatial_depth,
            num_heads=num_heads,
            head_dim=args.core_head_dim,
            ff_dim=args.core_ff_dim,
            ff_activation=args.core_ff_activation,
            mha_dropout=args.core_mha_dropout,
            ff_dropout=args.core_ff_dropout,
            drop_path=args.core_drop_path,
            use_rope=pos_encoding == 5,
            flash_attention=args.core_flash_attention == 1,
            is_causal=False,
            norm=args.core_norm,
            normalize_qk=normalize_qk,
            grad_checkpointing=args.grad_checkpointing,
        )
        self.temporal_transformer = Transformer(
            input_shape=(input_shape[0], emb_dim),
            reg_tokens=self.reg_tokens,
            depth=args.core_temporal_depth,
            num_heads=num_heads,
            head_dim=args.core_head_dim,
            ff_dim=args.core_ff_dim,
            ff_activation=args.core_ff_activation,
            mha_dropout=args.core_mha_dropout,
            ff_dropout=args.core_ff_dropout,
            drop_path=args.core_drop_path,
            use_rope=pos_encoding in (5, 6, 7),
            flash_attention=args.core_flash_attention == 1,
            is_causal=args.core_use_causal_attention,
            norm=args.core_norm,
            normalize_qk=normalize_qk,
            grad_checkpointing=args.grad_checkpointing,
        )

        self.output_shape = input_shape

    def compile(self):
        """Compile spatial and temporal transformer modules"""
        print(f"torch.compile spatial and temporal transformers in ViV1T")
        self.spatial_transformer = torch.compile(
            self.spatial_transformer,
            fullgraph=True,
        )
        self.temporal_transformer = torch.compile(
            self.temporal_transformer,
            fullgraph=True,
        )

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def add_reg_tokens(self, tokens: torch.Tensor):
        b, t, p, _ = tokens.shape
        # append spatial register tokens
        tokens = torch.cat(
            (tokens, repeat(self.reg_s_tokens, "r c -> b t r c", b=b, t=t)), dim=2
        )
        p += self.reg_tokens
        # append temporal register tokens
        tokens = torch.cat(
            (tokens, repeat(self.reg_t_tokens, "r c -> b r p c", b=b, p=p)), dim=1
        )
        return tokens

    def remove_reg_tokens(self, tokens: torch.Tensor):
        return tokens[:, : -self.reg_tokens, : -self.reg_tokens, :]

    def add_spatial_reg_tokens(self, tokens: torch.Tensor):
        b, t, p, _ = tokens.shape
        tokens = torch.cat(
            (tokens, repeat(self.reg_s_tokens, "r c -> b t r c", b=b, t=t)), dim=2
        )
        return tokens

    def remove_spatial_reg_tokens(self, tokens: torch.Tensor):
        return tokens[:, :, : -self.reg_tokens, :]

    def add_temporal_reg_tokens(self, tokens: torch.Tensor):
        b, t, p, _ = tokens.shape
        tokens = torch.cat(
            (tokens, repeat(self.reg_t_tokens, "r c -> b r p c", b=b, p=p)), dim=1
        )
        return tokens

    def remove_temporal_reg_tokens(self, tokens: torch.Tensor):
        return tokens[:, : -self.reg_tokens, :, :]

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        b, t, p, _ = outputs.shape

        if self.reg_tokens:
            outputs = self.add_spatial_reg_tokens(outputs)

        outputs = rearrange(outputs, "b t p c -> (b t) p c")
        outputs = self.spatial_transformer(
            outputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = rearrange(outputs, "(b t) p c -> b t p c", b=b)

        if self.reg_tokens:
            outputs = self.remove_spatial_reg_tokens(outputs)
            outputs = self.add_temporal_reg_tokens(outputs)

        outputs = rearrange(outputs, "b t p c -> (b p) t c")
        outputs = self.temporal_transformer(
            outputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = rearrange(outputs, "(b p) t c -> b t p c", b=b)

        if self.reg_tokens:
            outputs = self.remove_temporal_reg_tokens(outputs)

        return outputs


@register("vivit")
class ViViTCore(Core):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        verbose: int = 0,
    ):
        """
        Behavior mode (--core_behavior_mode)
            0: do not include behavior
            1: concat behavior with visual input
            2: concat behavior and pupil center with visual input
            3: cross attention with behavior and pupil center
        """
        super(ViViTCore, self).__init__(args, input_shape=input_shape, verbose=verbose)
        self.input_shape = input_shape
        self.behavior_mode = args.core_behavior_mode

        self.tokenizer = Tokenizer(
            input_shape=input_shape,
            behavior_mode=args.core_behavior_mode,
            patch_mode=args.core_patch_mode,
            spatial_patch_size=args.core_spatial_patch_size,
            spatial_patch_stride=args.core_spatial_patch_stride,
            temporal_patch_size=args.core_temporal_patch_size,
            temporal_patch_stride=args.core_temporal_patch_stride,
            pos_encoding=args.core_pos_encoding,
            emb_dim=args.core_emb_dim,
            dropout=args.core_p_dropout,
            max_frame=args.max_frame,
            pad_frame=bool(args.core_pad_frame),
            norm=args.core_norm,
        )

        self.vivit = ViViT(
            args,
            input_shape=self.tokenizer.output_shape,
            verbose=verbose,
        )

        new_h, new_w = self.tokenizer.new_shape
        self.rearrange = Rearrange("b t (h w) c -> b c t h w", h=new_h, w=new_w)
        self.activation = get_activation(args.core_activation)()

        self.output_shape = (
            args.core_emb_dim,
            self.tokenizer.output_shape[0],
            new_h,
            new_w,
        )

    def compile(self):
        self.vivit.compile()

    def regularizer(self):
        return self.vivit.regularizer()

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        outputs = self.tokenizer(
            outputs,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = self.vivit(
            outputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = self.rearrange(outputs)
        outputs = self.activation(outputs)
        return outputs
