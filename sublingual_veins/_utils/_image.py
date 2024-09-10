from typing import Tuple

import cv2
import numpy as np


def padding_resize(
    image: np.ndarray,
    size: Tuple[int, int] = (640, 640),
    stride: int = 32,
    full_padding: bool = True,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize and pad image to size with padding color.

    This function resizes the input image to the specified size and pads it with a specified color.
    The padding can be done in two ways: full padding, where the image is centered and padded on all sides,
    or partial padding, where the image is padded only on the right and bottom sides to match the stride
    of a convolutional neural network.

    Args:
        image (np.ndarray): The input image to resize.
        size (Tuple[int, int]): The target size to resize the image to. Defaults to (640, 640).
        stride (int): The stride of the convolutional neural network. Defaults to 32.
        full_padding (bool): If True, the image will be fully padded. If False, the image will be partially padded.
                             Defaults to True.
        color (Tuple[int, int, int]): The color used for padding. Defaults to (114, 114, 114).

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: A tuple containing the resized and padded image and the padding
    """
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)  # scale to resize
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # resized, no border
    dw = size[0] - new_w if full_padding else (stride - new_w % stride) % stride  # width padding
    dh = size[1] - new_h if full_padding else (stride - new_h % stride) % stride  # height padding
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return image, (dw, dh)
