import cv2
import numpy as np


def feather_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    """Generate a soft alpha mask by applying Gaussian blur to a binary mask.

    Args:
        mask (np.ndarray): Input binary mask, shape (H, W), values 0 or 255.
        radius (int): Gaussian kernel radius.

    Returns:
        np.ndarray: Alpha mask, shape (H, W), in range [0, 1].
    """
    blur = cv2.GaussianBlur(
        mask.astype(np.float32),
        (radius * 2 + 1, radius * 2 + 1),
        radius,
    )
    return np.clip(blur / 255.0, 0.0, 1.0)


def blend_image(src: np.ndarray, dst: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend two images using a per-pixel alpha mask.

    Args:
        src (np.ndarray): Foreground image, shape (H, W, 3).
        dst (np.ndarray): Background image, shape (H, W, 3).
        alpha (np.ndarray): Alpha mask, shape (H, W), in range [0, 1].

    Returns:
        np.ndarray: Blended image, shape (H, W, 3).
    """
    alpha_3 = alpha[..., None]
    out = src.astype(np.float32) * alpha_3 + dst.astype(np.float32) * (1 - alpha_3)
    return np.clip(out, 0, 255).astype(np.uint8)
