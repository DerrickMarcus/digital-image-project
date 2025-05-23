import cv2
import numpy as np

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
    radius: 引导滤波的邻域半径。小值，保留更多小细节，磨皮效果弱。大值，磨皮更彻底，但可能丢失自然纹理。
    eps: 引导滤波的平滑因子。小值，严格保留边缘，更像“局部均值”。大值，边缘被模糊，更像“全局低通”。
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


def slim_face(
    image: np.ndarray, face_mask: np.ndarray, gain: float = 0.9, k_size: int = 15
) -> np.ndarray:
    h, w = image.shape[:2]
    ys, xs = np.where(face_mask > 0)
    if xs.size == 0:
        return image

    # y_min = int(max(ys.min(), 0))
    # y_max = int(min(ys.max() + 1, h))

    x_min = int(max(xs.min(), w * 0.1))
    x_max = int(min(xs.max(), w * 0.9))
    x_center = (x_min + x_max) / 2.0
    w_face = x_max - x_min

    x_L = int(max(np.floor(x_center - w_face / 2.0 * gain), 0))
    x_R = int(min(np.ceil(x_center + w_face / 2.0 * gain), w))

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # map_x[y_min:y_max, :x_L] = np.linspace(0, x_min, x_L, False)[None, ...]
    # map_x[y_min:y_max, x_L:x_R] = np.linspace(x_min, x_max, x_R - x_L, False)[None, ...]
    # map_x[y_min:y_max, x_R:] = np.linspace(x_max, w, w - x_R, False)[None, ...]

    map_x[:, :x_L] = np.linspace(0, x_min, x_L, False)[None, ...]
    map_x[:, x_L:x_R] = np.linspace(x_min, x_max, x_R - x_L, False)[None, ...]
    map_x[:, x_R:] = np.linspace(x_max, w, w - x_R, False)[None, ...]

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_x = cv2.GaussianBlur(map_x, (k_size, k_size), 0)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def enlarge_eyes(
    image: np.ndarray,
    face_mask: np.ndarray,
    gain: float = 1.1,
    distance: tuple[float, float, float, float, float] = (0.1, 0.1, 0.1, 0.2, 0.5),
    k_size: int = 15,
) -> np.ndarray:
    h, w = image.shape[:2]
    ys, xs = np.where(face_mask > 0)
    if xs.size == 0:
        return image

    # 估计面部区域4个坐标
    x_face_min = int(max(xs.min(), w * 0.1))
    x_face_max = int(min(xs.max() + 1, w * 0.9))
    y_face_min = int(max(ys.min(), h * 0.1))
    y_face_max = int(min(ys.max() + 1, h * 0.9))

    w_face = x_face_max - x_face_min
    h_face = y_face_max - y_face_min

    # 估计现在眼睛区域的坐标
    w_eyes = (1 - sum(distance[0:3])) * w_face / 2.0
    x_left_eye_min = int(x_face_min + w_face * distance[0])
    x_left_eye_max = int(x_left_eye_min + w_eyes)
    x_left_eye_c = (x_left_eye_min + x_left_eye_max) / 2.0

    x_right_eye_max = int(x_face_max - w_face * distance[2])
    x_right_eye_min = int(x_right_eye_max - w_eyes)
    x_right_eye_c = (x_right_eye_min + x_right_eye_max) / 2.0

    y_eyes_min = int(y_face_min + h_face * distance[3])
    y_eyes_max = int(y_face_max - h_face * distance[4])
    y_eyes_c = (y_eyes_min + y_eyes_max) / 2.0
    h_eyes = y_eyes_max - y_eyes_min

    # 估计大眼后，边界处坐标对应原来眼睛内部的坐标
    x_left_new_min = int(
        max(np.floor(x_left_eye_c - w_eyes / 2.0 / gain), x_left_eye_min)
    )
    x_left_new_max = int(
        min(np.ceil(x_left_eye_c + w_eyes / 2.0 / gain), x_left_eye_max)
    )
    x_right_new_min = int(
        max(np.floor(x_right_eye_c - w_eyes / 2.0 / gain), x_right_eye_min)
    )
    x_right_new_max = int(
        min(np.ceil(x_right_eye_c + w_eyes / 2.0 / gain), x_right_eye_max)
    )
    y_new_min = int(max(np.floor(y_eyes_c - h_eyes / 2.0 / gain), y_eyes_min))
    y_new_max = int(min(np.ceil(y_eyes_c + h_eyes / 2.0 / gain), y_eyes_max))

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 计算坐标映射关系
    map_x[y_eyes_min:y_eyes_max, x_left_eye_min:x_left_eye_max] = np.linspace(
        x_left_new_min, x_left_new_max, x_left_eye_max - x_left_eye_min, False
    )[None, ...]
    map_x[y_eyes_min:y_eyes_max, x_right_eye_min:x_right_eye_max] = np.linspace(
        x_right_new_min, x_right_new_max, x_right_eye_max - x_right_eye_min, False
    )[None, ...]
    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_x = cv2.GaussianBlur(map_x, (k_size, k_size), 0)

    map_y[y_eyes_min:y_eyes_max, x_left_eye_min:x_left_eye_max] = np.linspace(
        y_new_min, y_new_max, y_eyes_max - y_eyes_min, False
    )[:, None]
    map_y[y_eyes_min:y_eyes_max, x_right_eye_min:x_right_eye_max] = np.linspace(
        y_new_min, y_new_max, y_eyes_max - y_eyes_min, False
    )[:, None]
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
    map_y = cv2.GaussianBlur(map_y, (k_size, k_size), 0)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
