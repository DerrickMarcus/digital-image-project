import cv2
import numpy as np

from detect_face import canny_edge, morphological
from preprocess import bilateral_filter, clahe_equalization, hsi_to_rgb, rgb_to_hsi


def whiten_face(
    image: np.ndarray,
    face_mask: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsi = rgb_to_hsi(rgb)

    mask = face_mask.astype(bool)
    i = (hsi[..., 2] * 255).astype(np.uint8)
    i_eq = clahe_equalization(i, clip_limit, tile_grid_size)
    i_eq = i_eq.astype(np.float32) / 255.0

    hsi[:, :, 2][mask] = i_eq[mask]
    hsi[:, :, 2] = np.clip(hsi[:, :, 2], 0.0, 1.0)

    whitened = cv2.cvtColor(hsi_to_rgb(hsi), cv2.COLOR_RGB2BGR)
    return whitened


def smooth_skin(
    image: np.ndarray,
    face_mask: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    smoothed = bilateral_filter(image, diameter, sigma_color, sigma_space)
    out = image.copy()
    mask = face_mask.astype(bool)
    out[mask] = smoothed[mask]
    return out


def squeeze_face(
    image: np.ndarray, face_mask: np.ndarray, gain: float = 0.9
) -> np.ndarray:
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    center_x = w / 2.0
    mask = face_mask.astype(bool)

    map_x[mask] = center_x + (map_x[mask] - center_x) * gain
    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = map_y.astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def enlarge_eyes(
    image: np.ndarray,
    face_mask: np.ndarray,
    gain: float = 1.1,
    edge_low: int = 50,
    edge_high: int = 150,
    k_size: int = 5,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = face_mask.astype(bool)
    roi = np.zeros_like(gray)
    roi[mask] = gray[mask]

    edges = canny_edge(roi, low_thresh=edge_low, high_thresh=edge_high)
    clean = morphological(edges, op="open", k_size=k_size)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eyes = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        eyes.append((int(x), int(y), int(r)))
    if not eyes:
        return image

    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    for cx, cy, r in eyes:
        dx = map_x - cx
        dy = map_y - cy
        mask_eye = dx * dx + dy * dy <= r * r
        map_x[mask_eye] = cx + dx[mask_eye] * gain
        map_y[mask_eye] = cy + dy[mask_eye] * gain

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
