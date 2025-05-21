import cv2
import numpy as np

from preprocess import hsi_to_rgb, rgb_to_hsi


def whiten_hsi_clahe(
    image: np.ndarray,
    face_mask: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    k_size: int = 21,
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsi = rgb_to_hsi(rgb)

    i = (hsi[..., 2] * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    i_eq = clahe.apply(i).astype(np.float32) / 255.0

    mask = face_mask.astype(bool)
    hsi[..., 2][mask] = i_eq[mask]

    result = cv2.cvtColor(hsi_to_rgb(hsi), cv2.COLOR_RGB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = image.astype(np.float32) * (1 - alpha) + result.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def whiten_hsi_gamma(
    image: np.ndarray, face_mask: np.ndarray, gamma: float = 0.8, k_size: int = 21
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsi = rgb_to_hsi(rgb)

    mask = face_mask.astype(bool)
    hsi[..., 2][mask] = np.clip(hsi[..., 2][mask] ** gamma, 0, 1)

    result = cv2.cvtColor(hsi_to_rgb(hsi), cv2.COLOR_RGB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = image.astype(np.float32) * (1 - alpha) + result.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def whiten_hsi_gain(
    image: np.ndarray,
    face_mask: np.ndarray,
    gain_i: float = 1.1,
    gain_s: float = 0.9,
    k_size: int = 21,
) -> np.ndarray:
    # s略微减小让肤色更粉嫩，略微增大让整体更柔和
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsi = rgb_to_hsi(rgb)

    mask = face_mask.astype(bool)
    hsi[..., 1][mask] = np.clip(hsi[..., 1][mask] * gain_s, 0, 1)
    hsi[..., 2][mask] = np.clip(hsi[..., 2][mask] * gain_i, 0, 1)

    result = cv2.cvtColor(hsi_to_rgb(hsi), cv2.COLOR_RGB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = image.astype(np.float32) * (1 - alpha) + result.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def whiten_lab_gain(
    image: np.ndarray,
    face_mask: np.ndarray,
    gain_l: float = 1.1,
    gain_a: float = 1.02,
    gain_b: float = 0.98,
    k_size: int = 21,
) -> np.ndarray:
    # a略微增大提升红色色度，b略微减小抑制黄色
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = face_mask.astype(bool)

    lab[..., 0][mask] = np.clip(lab[..., 0][mask] * gain_l, 0, 255)
    lab[..., 1][mask] = np.clip(lab[..., 1][mask] * gain_a, 0, 255)
    lab[..., 2][mask] = np.clip(lab[..., 2][mask] * gain_b, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = image.astype(np.float32) * (1 - alpha) + result.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def smooth_skin(
    image: np.ndarray,
    face_mask: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    smoothed = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    out = image.copy()
    mask = face_mask.astype(bool)
    out[mask] = smoothed[mask]
    return out


def smooth_skin_bilateral(
    image: np.ndarray,
    face_mask: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    iterations: int = 2,
    k_size: int = 11,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_smooth = lab[..., 0].copy()
    for _ in range(iterations):
        l_smooth = cv2.bilateralFilter(l_smooth, diameter, sigma_color, sigma_space)

    lab[..., 0] = l_smooth
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = (
        image.astype(np.float32) * (1.0 - alpha) + result.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
