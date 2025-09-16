"""
Attention readout from Pierzchlewicz et al. NeurIPS 2023

Code reference
- MultipleSharedMultiHeadAttention2d https://github.com/sinzlab/nnvision/blob/dfb28c4c0b21865905009db2ea47a8dae35f2f29/nnvision/models/readouts.py#L937
- SharedMultiHeadAttention2d https://github.com/KonstantinWilleke/neuralpredictors/blob/4ef51533f948970e511ee6061711db25e5e52217/neuralpredictors/layers/attention_readout.py#L363
"""

import math
from typing import Any

import numpy as np
import torch
from einops import einsum
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from viv1t.model.readout.readout import Readout
from viv1t.model.readout.readout import Readouts
from viv1t.model.readout.readout import register


class PositionalEncoding2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        learned: bool = False,
        width: int = None,
        height: int = None,
        stack_channels: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.stack_channels = stack_channels

        if width is None:
            width = height = max_len

        if learned:
            self.register_parameter(
                "twod_pe", nn.Parameter(torch.randn(d_model, (height * width)))
            )
        else:
            d_model = d_model // 2
            pe = torch.zeros(width, d_model)
            position = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            twod_pe = torch.zeros(height, width, d_model * 2)
            for xpos in range(height):
                for ypos in range(width):
                    twod_pe[xpos, ypos, :] = torch.cat(
                        [pe[0, xpos], pe[0, ypos]], dim=-1
                    )
            twod_pe = twod_pe.flatten(0, 1).T
            self.register_buffer("twod_pe", twod_pe)

    def forward(self, inputs: torch.Tensor):
        if len(inputs.shape) == 3:
            if not self.stack_channels:
                outputs = inputs + self.twod_pe[:, : inputs.size(-1)].unsqueeze(0)
            else:
                outputs = torch.hstack(
                    [
                        inputs,
                        self.twod_pe[:, : inputs.size(-1)]
                        .unsqueeze(0)
                        .repeat(inputs.size(0), 1, 1),
                    ]
                )
        elif len(inputs.shape) == 4:
            if not self.stack_channels:
                outputs = inputs + self.twod_pe[:, : inputs.size(-2)].unsqueeze(
                    0
                ).unsqueeze(-1)
            else:
                outputs = torch.hstack(
                    [
                        inputs,
                        self.twod_pe[:, : inputs.size(-2)]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .repeat(inputs.size(0), 1, 1, inputs.size(-1)),
                    ]
                )
        else:
            raise ValueError("Positional encoding only supports 3D and 4D tensors")
        outputs = self.dropout(outputs)
        return outputs


class SharedMultiHeadAttention2d(Readout):
    """
    A readout using a transformer layer with self attention.
    """

    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: np.ndarray | torch.Tensor,
        mean_responses: torch.Tensor = None,
        use_pos_enc: bool = True,
        learned_pos: bool = False,
        dropout_pos: float = 0.1,
        key_embedding: bool = False,
        value_embedding: bool = False,
        heads: int = 1,
        scale: bool = False,
        temperature: tuple[bool, float] = (False, 1.0),  # (learnable-per-neuron, value)
        layer_norm: bool = False,
        stack_pos_encoding: bool = False,
        n_pos_channels: int | None = None,
        embed_out_dim: int | None = None,
    ):

        super(SharedMultiHeadAttention2d, self).__init__(
            args,
            input_shape=input_shape,
            mouse_id=mouse_id,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
        self.bias_mode = args.readout_bias_mode
        self.heads = heads
        self.use_pos_enc = use_pos_enc
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        self.embed_out_dim = embed_out_dim

        c, t, w, h = input_shape

        self.initialize(
            stack_pos_encoding=stack_pos_encoding,
            n_pos_channels=n_pos_channels,
            mean_responses=mean_responses,
        )

        if scale:
            # prevent softmax gradients from vanishing (for large dim_head)
            scale = (c // self.heads) ** -0.5
        else:
            scale = 1.0
        self.register_buffer("scale", torch.tensor(scale))

        if temperature[0]:
            self.register_buffer("T", torch.tensor(temperature[1]))
        else:
            self.register_parameter(
                "T", nn.Parameter(torch.ones(self.num_neurons) * temperature[1])
            )

        if layer_norm:
            self.norm = nn.LayerNorm((c, w * h))
        else:
            self.norm = None

    def initialize_bias(self, mean_responses: torch.Tensor = None):
        match self.bias_mode:
            case 0:
                bias = None
            case 1:
                bias = torch.zeros(self.num_neurons)
            case 2:
                if mean_responses is None:
                    bias = torch.zeros(self.num_neurons)
                else:
                    bias = mean_responses
            case _:
                raise NotImplementedError(
                    f"--bias_mode {self.bias_mode} not implemented."
                )
        self.register_parameter("bias", None if bias is None else nn.Parameter(bias))

    def initialize(
        self,
        stack_pos_encoding: bool,
        n_pos_channels: int | None,
        mean_responses: np.ndarray | torch.Tensor | None = None,
    ):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        c, t, h, w = self.input_shape
        self.n_pos_channels = n_pos_channels
        self.stack_pos_encoding = stack_pos_encoding

        # only need when stacking
        c_query = c
        if n_pos_channels and stack_pos_encoding:
            c_query = c + n_pos_channels

        if self.value_embedding and self.embed_out_dim:
            features = torch.full(
                (1, self.embed_out_dim, self.num_neurons), fill_value=1 / c
            )
        else:
            features = torch.full((1, c, self.num_neurons), fill_value=1 / c)
        self.register_parameter("features", nn.Parameter(features))

        if self.key_embedding and self.embed_out_dim:
            neuron_query = torch.full(
                (1, self.embed_out_dim, self.num_neurons), fill_value=1 / c
            )
        else:
            neuron_query = torch.full((1, c_query, self.num_neurons), fill_value=1 / c)
        self.register_parameter("neuron_query", nn.Parameter(neuron_query))

        self.initialize_bias(mean_responses=mean_responses)

    def forward(self, key: torch.Tensor, value: torch.Tensor):
        """
        Propagates the input forwards through the readout
        Args:
            key, value: inputs, pre-computed from the parent class.
        Returns:
            y: neuronal activity
        """
        query = rearrange(self.neuron_query, "o (h d) n -> o h d n", h=self.heads)
        # compare neuron query with each spatial position (dot-product)
        # -> [Images, Heads, w*h, Neurons]
        dot = einsum(key, query, "i h d s, o h d n -> i h s n")
        dot = dot * self.scale / self.T
        # compute attention weights
        # -> [Images, Heads, w*h, Neurons]
        attention_weights = F.softmax(dot, dim=2)
        # compute average weighted with attention weights
        # -> [Images, Heads, Head_Dim, Neurons]
        outputs = einsum(value, attention_weights, "i h d s, i h s n -> i h d n")
        # -> [Images, Channels, Neurons]
        outputs = rearrange(outputs, "i h d n -> i (h d) n")
        # -> [Images, Neurons]
        outputs = einsum(outputs, self.features, "i c n, o c n -> i n")
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs


@register("attention")
class AttentionReadouts(Readouts):
    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        neuron_coordinates: dict[str, np.ndarray | torch.Tensor],
        mean_responses: dict[str, np.ndarray | torch.Tensor] | None = None,
        use_pos_enc: bool = True,
        learned_pos: bool = False,
        heads: int = 1,
        scale: bool = True,
        key_embedding: bool = True,
        value_embedding: bool = True,
        temperature: tuple[bool, float] = (True, 1.0),  # (learnable-per-neuron, value)
        dropout_pos: float = 0.1,
        layer_norm: bool = False,
        stack_pos_encoding: bool = False,
        n_pos_channels: int | None = None,
    ):
        super(AttentionReadouts, self).__init__(
            args,
            input_shape=input_shape,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
            readout=SharedMultiHeadAttention2d,
        )
        self.bias_mode = args.readout_bias_mode
        c, t, h, w = self.input_shape

        self.heads = heads
        self.key_embedding = key_embedding
        self.value_embedding = value_embedding
        self.use_pos_enc = use_pos_enc

        embed_out_dim = args.readout_emb_dim

        if n_pos_channels and stack_pos_encoding:
            c = c + n_pos_channels
        c_out = c if not embed_out_dim else embed_out_dim

        d_model = n_pos_channels if n_pos_channels else c
        if self.use_pos_enc:
            self.position_embedding = PositionalEncoding2D(
                d_model=d_model,
                width=w,
                height=h,
                learned=learned_pos,
                dropout=dropout_pos,
                stack_channels=stack_pos_encoding,
            )

        if layer_norm:
            self.norm = nn.LayerNorm((c, w * h))
        else:
            self.norm = None

        if self.key_embedding and self.value_embedding:
            self.to_kv = nn.Linear(c, c_out * 2, bias=False)
        elif self.key_embedding:
            self.to_key = nn.Linear(c, c_out, bias=False)

        # super init to get the _module attribute
        for mouse_id in neuron_coordinates.keys():
            self.add_module(
                name=mouse_id,
                module=SharedMultiHeadAttention2d(
                    args=args,
                    input_shape=input_shape,
                    mouse_id=mouse_id,
                    neuron_coordinates=neuron_coordinates[mouse_id],
                    mean_responses=(
                        None if mean_responses is None else mean_responses[mouse_id]
                    ),
                    use_pos_enc=False,
                    key_embedding=key_embedding,
                    value_embedding=value_embedding,
                    heads=heads,
                    scale=scale,
                    temperature=temperature,  # (learnable-per-neuron, value)
                    layer_norm=layer_norm,
                    stack_pos_encoding=stack_pos_encoding,
                    n_pos_channels=n_pos_channels,
                    embed_out_dim=embed_out_dim,
                ),
            )

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        b, c, t, w, h = inputs.size()
        outputs = rearrange(inputs, "b c t h w -> (b t) c (h w)")

        if self.use_pos_enc:
            x_embed = self.position_embedding(outputs)  # -> [Images, Channels, w*h]
        else:
            x_embed = outputs

        if self.norm is not None:
            x_embed = self.norm(x_embed)

        if self.key_embedding and self.value_embedding:
            key, value = self.to_kv(rearrange(x_embed, "b c s -> (b s) c")).chunk(
                2, dim=-1
            )
            key = rearrange(
                key, "(b s) (h d) -> b h d s", h=self.heads, b=outputs.size(0)
            )
            value = rearrange(
                value, "(b s) (h d) -> b h d s", h=self.heads, b=outputs.size(0)
            )
        elif self.key_embedding:
            key = self.to_key(rearrange(x_embed, "b c s -> (b s) c"))
            key = rearrange(
                key, "(b s) (h d) -> b h d s", h=self.heads, b=outputs.size(0)
            )
            value = rearrange(outputs, "b (h d) s -> b h d s", h=self.heads)
        else:
            key = rearrange(x_embed, "b (h d) s -> b h d s", h=self.heads)
            value = rearrange(outputs, "b (h d) s -> b h d s", h=self.heads)

        outputs = self[mouse_id](key, value)
        outputs = rearrange(outputs, "(b t) n -> b n t", b=b, t=t)
        return outputs
