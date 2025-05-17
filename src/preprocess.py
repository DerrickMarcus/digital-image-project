import cv2
import numpy as np


def rgb_to_hsi(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSI color space.

    Args:
        rgb (np.ndarray): Input RGB image of shape (H, W, 3), in range [0, 255].

    Returns:
        np.ndarray: HSI image of shape (H, W, 3), in range [0, 1].
    """

    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    i = (r + g + b) / 3.0

    s = np.zeros_like(i)
    min_rgb = np.minimum(np.minimum(r, g), b)
    non_zero = i > 0
    s[non_zero] = 1 - min_rgb[non_zero] / i[non_zero]

    h = np.zeros_like(i)
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    non_zero = denominator > 0
    h[non_zero] = np.arccos(numerator[non_zero] / denominator[non_zero])
    h[b > g] = 2 * np.pi - h[b > g]
    h = h / (2 * np.pi)

    hsi = np.stack([h, s, i], axis=-1)
    return hsi


def hsi_to_rgb(hsi: np.ndarray) -> np.ndarray:
    """Convert HSI image to RGB color space.

    Args:
        hsi (np.ndarray): Input HSI image of shape (H, W, 3), in range [0, 1].

    Returns:
        np.ndarray: RGB image of shape (H, W, 3), in range [0, 255].
    """

    h, s, i = hsi[:, :, 0], hsi[:, :, 1], hsi[:, :, 2]
    h = h * 2 * np.pi
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)

    area_1 = h < 2 * np.pi / 3
    b[area_1] = i[area_1] * (1 - s[area_1])
    r[area_1] = i[area_1] * (
        1 + s[area_1] * np.cos(h[area_1]) / np.cos(np.pi / 3 - h[area_1])
    )
    g[area_1] = 3 * i[area_1] - (r[area_1] + b[area_1])

    area_2 = (2 * np.pi / 3 <= h) & (h < 4 * np.pi / 3)
    r[area_2] = i[area_2] * (1 - s[area_2])
    g[area_2] = i[area_2] * (
        1 + s[area_2] * np.cos(h[area_2] - 2 * np.pi / 3) / np.cos(np.pi - h[area_2])
    )
    b[area_2] = 3 * i[area_2] - (r[area_2] + g[area_2])

    area_3 = 4 * np.pi / 3 <= h
    g[area_3] = i[area_3] * (1 - s[area_3])
    b[area_3] = i[area_3] * (
        1
        + s[area_3]
        * np.cos(h[area_3] - 4 * np.pi / 3)
        / np.cos(5 * np.pi / 3 - h[area_3])
    )
    r[area_3] = 3 * i[area_3] - (b[area_3] + g[area_3])

    rgb = np.stack([r, g, b], axis=-1)
    rgb = (np.clip(rgb * 255.0, 0, 255)).astype(np.uint8)
    return rgb


def median_filter(image: np.ndarray, k_size: int) -> np.ndarray:
    return cv2.medianBlur(image, k_size)


def gaussian_filter(
    image: np.ndarray, k_size: int = 5, sigma: float = 1.0
) -> np.ndarray:
    return cv2.GaussianBlur(image, (k_size, k_size), sigma)


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Apply bilateral filtering.

    Args:
        image (np.ndarray): Input image.
        diameter (int, optional): Diameter of the pixel neighborhood. Defaults to 5.
        sigma_color (float, optional): Filter sigma in color space. Defaults to 75.
        sigma_space (float, optional): Filter sigma in coordinate space. Defaults to 75.

    Returns:
        np.ndarray: Filtered image.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)


def clahe_equalization(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)


def laplacian_sharpen(image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    lap = cv2.Laplacian(image, cv2.CV_32F)
    sharpened = image.astype(np.float32) + alpha * lap
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    ksize: tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, ksize, sigma)
    mask = cv2.subtract(image, blurred)
    sharpened = cv2.addWeighted(image, 1.0, mask, amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
