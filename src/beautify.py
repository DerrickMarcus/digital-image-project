import cv2
import numpy as np

from detect_face import canny_edge
from preprocess import hsi_to_rgb, rgb_to_hsi


def whiten_lab_clahe(
    image: np.ndarray,
    face_mask: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    k_size: int = 21,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]

    # 对全局图像的L通道做均衡化
    # mask = face_mask.astype(bool)
    # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # L_eq = clahe.apply(L)
    # lab[..., 0][mask] = L_eq[mask]

    # 只对人脸区域的L通道做均衡化
    mask = face_mask.astype(bool)
    L_face = L.copy()
    L_face[~mask] = int(np.median(L[mask]))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_eq = clahe.apply(L_face)
    lab[..., 0][mask] = L_eq[mask]

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = (
        image.astype(np.float32) * (1.0 - alpha) + result.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


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


def smooth_skin_guided(
    image: np.ndarray,
    face_mask: np.ndarray,
    radius: int = 8,
    eps: float = 1e-2,
    k_size: int = 11,
) -> np.ndarray:
    """
    radius: 引导滤波的邻域半径。
    eps: 引导滤波的平滑因子（越小保留边缘越强）。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].copy()

    L_f = L.astype(np.float32) / 255.0
    L_guided = cv2.ximgproc.guidedFilter(guide=L_f, src=L_f, radius=radius, eps=eps)
    L_u = (L_guided * 255.0).astype(np.uint8)

    lab[..., 0] = L_u
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    alpha = cv2.GaussianBlur(
        (face_mask / 255.0).astype(np.float32), (k_size, k_size), 0
    )[..., None]
    blended = (
        image.astype(np.float32) * (1.0 - alpha) + result.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

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
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
