import torch

from utils import normalize_kernel2d
import torch.nn.functional as F

def get_sobel_kernel_3x3(device: torch.device  = None, dtype: torch.dtype  = None) -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype)

def get_sobel_kernel2d(device: torch.device  = None, dtype: torch.dtype  = torch.float) -> torch.Tensor:
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])

def get_spatial_gradient_kernel2d(
    mode: str, order: int, *, device: torch.device = None, dtype: torch.dtype = torch.float
) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients, using one of the following operators:

    sobel, diff.
    """
    assert mode == 'sobel' and order == 1
    kernel: torch.Tensor = get_sobel_kernel2d(device='cpu', dtype=dtype)
    return kernel


def spatial_gradient(input: torch.Tensor, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> torch.Tensor:
    r"""Compute the first order image derivative in both x and y using a Sobel operator.

    .. image:: _static/img/spatial_gradient.png

    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_edges.html>`__.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """

    # allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode, order, device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: torch.Tensor = F.pad(input.reshape(c, 1, h, w), spatial_pad, 'replicate')
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(c, out_channels, h, w)


