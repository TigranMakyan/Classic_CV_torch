import torch
import torch.nn.functional as F

import math

from utils import rgb_to_grayscale, get_canny_nms_kernel, get_hysteresis_kernel
from gaussian import gaussian_blur2d
from sobel import spatial_gradient

import sys
sys.path.insert(0, '.')

def canny(
    input: torch.Tensor,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: tuple[float, float] = (1, 1),
    hysteresis: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Find edges of the input image and filters them using the Canny algorithm.

    .. image:: _static/img/canny.png

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       canny.html>`__.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    device = input.device
    dtype = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # Gaussian filter
    blurred: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)
    print(f'Blurred image shape is {blurred.shape}')

    # Compute the gradients
    gradients: torch.Tensor = spatial_gradient(blurred, normalized=False)
    print(f'gradients shape is {gradients.shape}')

    # Unpack the edges
    gx: torch.Tensor = gradients[:, :, 0]
    gy: torch.Tensor = gradients[:, :, 1]
    print(f'Unpack edges: gx : {gx.shape}, gy: {gy.shape}')
    # Compute gradient magnitude and angle
    magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    print(f'Magnitude shape : {magnitude.shape}')
    angle: torch.Tensor = torch.atan2(gy, gx)

    # Radians to Degrees
    angle = 180.0 * angle / math.pi

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels: torch.Tensor = get_canny_nms_kernel(device, dtype)
    print(f'nms_kernels shape is {nms_kernels.shape}')
    nms_magnitude: torch.Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # Get the indices for both directions
    positive_idx: torch.Tensor = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx: torch.Tensor = ((angle / 45) + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive: torch.Tensor = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative: torch.Tensor = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered: torch.Tensor = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max: torch.Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

    magnitude = magnitude * is_max

    # Threshold
    edges: torch.Tensor = F.threshold(magnitude, low_threshold, 0.0)

    low: torch.Tensor = magnitude > low_threshold
    high: torch.Tensor = magnitude > high_threshold

    edges = low * 0.5 + high * 0.5
    edges = edges.to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old: torch.Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels: torch.Tensor = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            weak: torch.Tensor = (edges == 0.5).float()
            strong: torch.Tensor = (edges == 1).float()

            hysteresis_magnitude: torch.Tensor = F.conv2d(
                edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

        edges = hysteresis_magnitude

    return magnitude, edges
