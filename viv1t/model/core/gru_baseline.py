"""
Code reference
- https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/cores/conv2d.py
- https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/hermite.py
"""

import math
from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from typing import Any
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from numpy import pi
from numpy.polynomial.polynomial import polyval
from scipy.special import gamma
from torch import nn

from viv1t.model.core.core import Core
from viv1t.model.core.core import register
from viv1t.model.core.gru import GRU
from viv1t.model.core.stacked2d import Stacked2d


def factorial(a: Union[int, float, np.ndarray]):
    assert isinstance(a, int) or a.is_integer()
    return math.factorial(int(a))


def hermite_coefficients(mu, nu):
    """Generate coefficients of 2D Hermite functions"""
    nur = np.arange(nu + 1)
    num = gamma(mu + nu + 1) * gamma(nu + 1) * ((-2) ** (nu - nur))
    denom = gamma(mu + 1 + nur) * gamma(1 + nur) * gamma(nu + 1 - nur)
    return num / denom


def hermite_2d(N, npts, xvalmax=None):
    """Generate 2D Hermite function basis

    Arguments:
    N           -- the maximum rank.
    npts        -- the number of points in x and y

    Keyword arguments:
    xvalmax     -- the maximum x and y value (default: 2.5 * sqrt(N))

    Returns:
    H           -- Basis set of size N*(N+1)/2 x npts x npts
    desc        -- List of descriptors specifying for each
                   basis function whether it is:
                        'z': rotationally symmetric
                        'r': real part of quadrature pair
                        'i': imaginary part of quadrature pair

    """
    xvalmax = xvalmax or 2.5 * np.sqrt(N)
    ranks = range(N)

    # Gaussian envelope
    xvalmax *= 1 - 1 / npts
    xvals = np.linspace(-xvalmax, xvalmax, npts, endpoint=True)[..., None]

    gxv = np.exp(-(xvals**2) / 4)
    gaussian = np.dot(gxv, gxv.T)

    # Hermite polynomials
    mu = np.array([])
    nu = np.array([])
    desc = []
    for i, rank in enumerate(ranks):
        muadd = np.sort(np.abs(np.arange(-rank, rank + 0.1, 2)))
        mu = np.hstack([mu, muadd])
        nu = np.hstack([nu, (rank - muadd) / 2])
        if not (rank % 2):
            desc.append("z")
        desc += ["r", "i"] * int(np.floor((rank + 1) / 2))

    theta = np.arctan2(xvals, xvals.T)
    radsq = xvals**2 + xvals.T**2
    nbases = mu.size
    H = np.zeros([nbases, npts, npts])
    for i, (mui, nui, desci) in enumerate(zip(mu, nu, desc)):
        radvals = polyval(radsq, hermite_coefficients(mui, nui))
        basis = gaussian * (radsq ** (mui / 2)) * radvals * np.exp(1j * mui * theta)
        basis /= np.sqrt(
            2 ** (mui + 2 * nui) * pi * factorial(mui + nui) * factorial(nui)
        )
        if desci == "z":
            H[i] = basis.real / np.sqrt(2)
        elif desci == "r":
            H[i] = basis.real
        elif desci == "i":
            H[i] = basis.imag

    # normalize
    return H / np.sqrt(np.sum(H**2, axis=(1, 2), keepdims=True)), desc, mu


def rotation_matrix(desc, mu, angle):
    R = np.zeros((len(desc), len(desc)))
    for i, (d, m) in enumerate(zip(desc, mu)):
        if d == "r":
            Rc = np.array(
                [
                    [np.cos(m * angle), np.sin(m * angle)],
                    [-np.sin(m * angle), np.cos(m * angle)],
                ]
            )
            R[i : i + 2, i : i + 2] = Rc
        elif d == "z":
            R[i, i] = 1
    return R


def downsample_weights(weights: torch.Tensor, factor: int = 2) -> torch.Tensor:
    w = 0
    for i in range(factor):
        for j in range(factor):
            w += weights[i::factor, j::factor]
    return w


class RotateHermite(nn.Module):
    def __init__(
        self,
        filter_size: int,
        upsampling: int,
        num_rotations: int,
        first_layer: bool,
    ):
        super(RotateHermite, self).__init__()

        H, desc, mu = hermite_2d(
            filter_size, filter_size * upsampling, 2 * np.sqrt(filter_size)
        )

        self.H = nn.Parameter(torch.tensor(H, dtype=torch.float32), requires_grad=False)

        angles = [i * 2 * pi / num_rotations for i in range(num_rotations)]
        Rs = [
            torch.tensor(rotation_matrix(desc, mu, angle), dtype=torch.float32)
            for angle in angles
        ]

        self.Rs = nn.ParameterList([nn.Parameter(R, requires_grad=False) for R in Rs])

        self.num_rotations = num_rotations
        self.first_layer = first_layer

    def forward(self, coeffs: torch.Tensor):
        num_inputs_total = coeffs.shape[1]  # num_coeffs, num_inputs_total, num_outputs
        num_inputs = num_inputs_total // self.num_rotations

        weights_rotated = []
        for i, R in enumerate(self.Rs):
            coeffs_rotated = torch.tensordot(R, coeffs, dims=([1], [0]))
            w = torch.tensordot(self.H, coeffs_rotated, dims=[[0], [0]])
            if i and not self.first_layer:
                shift = num_inputs_total - i * num_inputs
                w = torch.cat([w[:, :, shift:, :], w[:, :, :shift, :]], dim=2)
            weights_rotated.append(w)
        weights_all_rotations = torch.cat(weights_rotated, dim=3)

        return weights_all_rotations


class HermiteConv2D(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        filter_size: int,
        padding: int,
        stride: int,
        num_rotations: int,
        upsampling: int,
        first_layer: bool,
    ):
        super(HermiteConv2D, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.padding = padding
        self.stride = stride
        self.upsampling = upsampling
        self.n_coeffs = filter_size * (filter_size + 1) // 2

        coeffs = nn.Parameter(
            torch.Tensor(self.n_coeffs, self.input_features, self.output_features)
        )
        self.coeffs = coeffs

        self.rotate_hermite = RotateHermite(
            filter_size=filter_size,
            upsampling=upsampling,
            num_rotations=num_rotations,
            first_layer=first_layer,
        )

    @property
    def weights_all_rotations(self):
        weights_all_rotations = self.rotate_hermite(self.coeffs)
        weights_all_rotations = downsample_weights(
            weights_all_rotations, self.upsampling
        )
        weights_all_rotations = weights_all_rotations.permute(3, 2, 0, 1)
        return weights_all_rotations

    def forward(self, inputs: torch.Tensor):
        return F.conv2d(
            input=inputs,
            weight=self.weights_all_rotations,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )


class RotationEquivariantBatchNorm2D(nn.Module):
    def __init__(
        self,
        num_features,
        num_rotations,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        self.num_features = num_features
        self.num_rotations = num_rotations
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.batch_norm = nn.BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input):
        s = input.shape
        input = self.reshape(
            input, s
        )  # rotations will share BN parameters for each channel
        output = self.batch_norm(input)
        output = self.inv_reshape(output, s)
        return output

    def reshape(self, x, s):
        x = x.view(s[0], self.num_rotations, self.num_features, s[2], s[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(s[0], self.num_features, self.num_rotations * s[2] * s[3])
        return x

    def inv_reshape(self, x, s):
        x = x.view(s[0], self.num_features, self.num_rotations, s[2], s[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(s[0], self.num_rotations * self.num_features, s[2], s[3])
        return x


class RotationEquivariantBias2DLayer(nn.Module):
    def __init__(self, channels, num_rotations, initial=0, **kwargs):
        super().__init__(**kwargs)

        self.num_features = channels
        self.num_rotations = num_rotations
        self.bias = torch.nn.Parameter(
            torch.empty((1, 1, channels, 1, 1)).fill_(initial)
        )

    def forward(self, x):
        s = x.shape
        x = x.view(s[0], self.num_rotations, self.num_features, s[2], s[3])
        x = x + self.bias
        return x.view(*s)


class RotationEquivariantScale2DLayer(nn.Module):
    def __init__(self, channels, num_rotations, initial=1, **kwargs):
        super().__init__(**kwargs)

        self.num_features = channels
        self.num_rotations = num_rotations
        self.scale = torch.nn.Parameter(
            torch.empty((1, 1, channels, 1, 1)).fill_(initial)
        )

    def forward(self, x):
        s = x.shape
        x = x.view(s[0], self.num_rotations, self.num_features, s[2], s[3])
        x = x * self.scale
        return x.view(*s)


class RotationEquivariantStacked2d(Stacked2d):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int],
        upsampling: int = 2,
        rot_eq_batch_norm: bool = True,
        init_std: float = 0.1,
    ):
        self.num_rotations = args.core_rotations
        self.upsampling = upsampling
        self.rot_eq_batch_norm = rot_eq_batch_norm
        self.init_std = init_std
        super(RotationEquivariantStacked2d, self).__init__(
            args, input_shape=input_shape
        )

        self.output_shape = (
            self.hidden_channels[-1] * self.num_rotations,
            args.crop_frame,
            input_shape[2] - self.input_kern + 1,
            input_shape[3] - self.input_kern + 1,
        )

    def set_batchnorm_type(self):
        if not self.rot_eq_batch_norm:
            super().set_batchnorm_type()
        else:
            self.batchnorm_layer_cls = partial(
                RotationEquivariantBatchNorm2D, num_rotations=self.num_rotations
            )
            self.bias_layer_cls = partial(
                RotationEquivariantBias2DLayer, num_rotations=self.num_rotations
            )
            self.scale_layer_cls = partial(
                RotationEquivariantScale2DLayer, num_rotations=self.num_rotations
            )

    def add_first_layer(self):
        layer = OrderedDict()
        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.input_shape[0],
            output_features=self.hidden_channels[0],
            num_rotations=self.num_rotations,
            upsampling=self.upsampling,
            filter_size=self.input_kern,
            stride=self.stride,
            padding=self.input_kern // 2 if self.pad_input else 0,
            first_layer=True,
        )
        self.add_bn_layer(layer, self.hidden_channels[0])
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()

            if self.hidden_padding is None:
                self.hidden_padding = self.hidden_kern[l - 1] // 2

            layer["hermite_conv"] = HermiteConv2D(
                input_features=self.hidden_channels[l - 1] * self.num_rotations,
                output_features=self.hidden_channels[l],
                num_rotations=self.num_rotations,
                upsampling=self.upsampling,
                filter_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                first_layer=False,
            )
            self.add_bn_layer(layer, self.hidden_channels[l])
            self.add_activation(layer)
            self.features.add_module(f"layer{l}", nn.Sequential(layer))

    def initialize(self):
        self.apply(self.init_conv_hermite)

    def init_conv_hermite(self, m):
        if isinstance(m, HermiteConv2D):
            nn.init.normal_(m.coeffs.data, std=self.init_std)

    def laplace(self):
        return self._input_weights_regularizer(
            self.features[0].hermite_conv.weights_all_rotations, avg=self.use_avg_reg
        )

    def group_sparsity(self):
        ret = 0
        for feature in self.features[1:]:
            ret += (
                feature.hermite_conv.weights_all_rotations.pow(2)
                .sum(3, keepdim=True)
                .sum(2, keepdim=True)
                .sqrt()
                .mean()
            )
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def forward(self, inputs: torch.Tensor):
        outputs = []
        for l, feat in enumerate(self.features):
            inputs = feat(inputs)
            outputs.append(inputs)
        return torch.cat([outputs[i] for i in self.stack], dim=1)


@register("gru_baseline")
class RotationEquivariantCNNGRUCore(Core):
    def __init__(
        self, args: Any, input_shape: Tuple[int, int, int, int], verbose: int = 0
    ):
        """
        Behavior mode (--core_behavior_mode)
            0: do not include behavior
            1: concat behavior with visual input
            2: concat behavior and pupil center with visual input
        """
        super(RotationEquivariantCNNGRUCore, self).__init__(
            args,
            input_shape=input_shape,
            verbose=verbose,
        )
        self.input_shape = input_shape
        self.behavior_mode = args.core_behavior_mode
        input_shape = list(input_shape)
        match self.behavior_mode:
            case 0:
                pass
            case 1:
                input_shape[0] += args.input_shapes["behavior"][0]
            case 2:
                input_shape[0] += (
                    args.input_shapes["behavior"][0]
                    + args.input_shapes["pupil_center"][0]
                )
            case _:
                raise NotImplementedError(f"--behavior_mode {self.behavior_mode}")
        self.cnn = RotationEquivariantStacked2d(args, input_shape=tuple(input_shape))
        output_shape = self.cnn.output_shape
        self.gru = GRU(args, input_shape=output_shape)
        self.output_shape = self.gru.output_shape

    def regularizer(self):
        cnn_reg = self.cnn.regularizer()
        gru_reg = self.gru.regularizer()
        return cnn_reg + gru_reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, _, t, h, w = inputs.shape
        outputs = rearrange(inputs, "b c t h w -> (b t) c h w")
        match self.behavior_mode:
            case 1:
                behaviors = rearrange(behaviors, "b d t -> (b t) d")
                behaviors = repeat(behaviors, "b d -> b d h w", h=h, w=w)
                outputs = torch.concat((outputs, behaviors), dim=1)
            case 2:
                behaviors = rearrange(behaviors, "b d t -> (b t) d")
                behaviors = repeat(behaviors, "b d -> b d h w", h=h, w=w)
                pupil_centers = rearrange(pupil_centers, "b d t -> (b t) d")
                pupil_centers = repeat(pupil_centers, "b d -> b d h w", h=h, w=w)
                outputs = torch.concat((outputs, behaviors, pupil_centers), dim=1)
        outputs = self.cnn(outputs)
        outputs = rearrange(outputs, "(b t) c h w -> b c t h w", b=b, t=t)
        outputs = self.gru(outputs)
        return outputs
