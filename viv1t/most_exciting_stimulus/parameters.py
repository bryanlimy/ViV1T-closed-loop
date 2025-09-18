import torch


def get_pixel_param(
    shape: tuple[int, int, int, int],
    num_frames: int,
    sd: float = 0.01,
    param: torch.Tensor | None = None,
) -> torch.Tensor:
    if param is None:
        param = sd * torch.randn(size=(shape[0], num_frames, shape[2], shape[3]))
    return param


def get_domain_filter(size: int, cutoff: float) -> torch.Tensor:
    """
    Return a low-pass filter in the frequency domain which retains the
    cutoff% of the low frequency components.
    """
    assert cutoff >= 0 and cutoff <= 1
    freq = torch.fft.fftfreq(size)
    magnitude = torch.sqrt(freq**2)
    cutoff_freq = cutoff * magnitude.max()
    return (magnitude <= cutoff_freq).to(torch.float32)


def get_fft_param(
    shape: tuple[int, int, int, int],
    num_frames: int,
    method: str,
    spatial_cutoff: float | None = None,
    temporal_cutoff: float | None = None,
    sd: float = 0.01,
    norm: str = "ortho",
    param: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if param is None:
        param = sd * torch.randn(size=(shape[0], num_frames, shape[2], shape[3]))
    param = torch.fft.fftn(param, dim=(1, 2, 3), norm=norm)
    match method:
        case "kl":
            scale = torch.ones_like(param)
        case "cutoff":
            t_filter = get_domain_filter(size=num_frames, cutoff=temporal_cutoff)
            if num_frames == 1:
                t_filter = torch.ones_like(t_filter)
            h_filter = get_domain_filter(size=shape[2], cutoff=spatial_cutoff)
            w_filter = get_domain_filter(size=shape[3], cutoff=spatial_cutoff)
            scale = (
                t_filter[:, None, None]
                * h_filter[None, :, None]
                * w_filter[None, None, :]
            )
        case _:
            raise NotImplementedError(f"Unknown param method {method}")
    return param, scale


def get_param(
    shape: tuple[int, int, int, int],
    num_frames: int,
    method: str,
    spatial_cutoff: float | None = None,
    temporal_cutoff: float | None = None,
    sd: float = 0.01,
    norm: str = "ortho",
    device: torch.device = torch.device("cpu"),
    param: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    match method:
        case "pixel":
            param = get_pixel_param(
                shape=shape,
                num_frames=num_frames,
                param=param,
            )
            scale = torch.ones_like(param)
        case "kl" | "cutoff":
            param, scale = get_fft_param(
                shape=shape,
                num_frames=num_frames,
                method=method,
                spatial_cutoff=spatial_cutoff,
                temporal_cutoff=temporal_cutoff,
                sd=sd,
                norm=norm,
                param=param,
            )
        case _:
            raise NotImplementedError(f"Unknown param method {method}")
    param = param.to(device).requires_grad_(True)
    scale = scale.to(device)
    return param, scale
