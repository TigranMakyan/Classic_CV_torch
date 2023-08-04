from typing import Optional, List
import torch

def rgb_to_grayscale(image: torch.Tensor, rgb_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images 
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def get_canny_nms_kernel(device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )

def get_hysteresis_kernel(device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    return torch.tensor(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        device=device,
        dtype=dtype,
    )
