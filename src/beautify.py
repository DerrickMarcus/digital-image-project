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


def slim_face(
    image: np.ndarray, face_mask: np.ndarray, gain: float = 0.9, k_size: int = 15
) -> np.ndarray:
    h, w = image.shape[:2]
    _, xs = np.where(face_mask > 0)
    if xs.size == 0:
        return image

    x_min = max(int(xs.min()), w / 6)
    x_max = min(int(xs.max()), w * 5 / 6)
    x_center = (x_min + x_max) / 2.0
    w_face = x_max - x_min
    x_L = int(np.floor(x_center - w_face / 2.0 * gain))
    x_R = int(np.ceil(x_center + w_face / 2.0 * gain))

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    map_x[:, :x_L] = np.linspace(0, x_min - 1, x_L)[None, :]
    map_x[:, x_L : x_R + 1] = np.linspace(x_min, x_max, x_R - x_L + 1)[None, :]
    map_x[:, x_R + 1 :] = np.linspace(x_max + 1, w - 1, w - x_R - 1)[None, :]

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_x = cv2.GaussianBlur(map_x, (k_size, k_size), 0)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def get_eyes_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    mask = (
        cv2.inRange(hsv, (0, 40, 40), (25, 120, 240))
        | cv2.inRange(hsv, (155, 40, 40), (179, 120, 240))
        | cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        # | cv2.inRange(lab[..., 0], 0, 128)
        # | cv2.inRange(lab, (0, 128 - 5, 128 - 5), (255, 128 - 5, 128 + 20))
    )

    cr = ycrcb[:, :, 1]
    _, mask_cr = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask | mask_cr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final, [c], -1, 255, thickness=-1)

    final = mask

    # 膨胀操作使得轮廓扩大
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final = cv2.dilate(final, kernel, iterations=1)

    final = cv2.GaussianBlur(final, (51, 51), 0)
    _, final = cv2.threshold(final, 128, 255, cv2.THRESH_BINARY)

    return final


def get_eyes_mask_hough(
    image: np.ndarray,
    face_mask: np.ndarray,
    dp: float = 1.2,
    min_dist_ratio: float = 0.25,
    canny_thresh1: int = 50,
    canny_thresh2: int = 150,
    hough_param2: int = 30,
    radius_ratio: tuple[float, float] = (0.05, 0.15),
) -> np.ndarray:
    """
    基于 Canny + HoughCircles 检测双眼，并返回眼睛区域二值掩膜。

    Args:
        image: BGR 原图, shape (H, W, 3).
        face_mask: 人脸区域二值掩膜, shape (H, W), 0 or 255.
        dp: Hough 圆检测累加器分辨率比 (image / dp).
        min_dist_ratio: 两眼最小距离占脸宽比例, 默认 0.25.
        canny_thresh1/2: Canny 边缘检测低/高阈值.
        hough_param2: HoughCircles 参数2（累加器阈值，越大越少检测）。
        radius_ratio: (minRadius_ratio, maxRadius_ratio) 相对于脸宽。

    Returns:
        eyes_mask: 0/255 二值图, 只有检测到的左右眼圆区域为白。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) 在 face_mask ROI 内裁剪上半脸
    ys, xs = np.where(face_mask > 0)
    if xs.size == 0:
        return np.zeros((h, w), np.uint8)
    y_min, y_max = ys.min(), ys.max()
    y_mid = int((y_min + y_max) / 2)
    roi_gray = gray[y_min:y_mid, :]
    roi_w = roi_gray.shape[1]
    # 2) 边缘检测
    edges = cv2.Canny(roi_gray, canny_thresh1, canny_thresh2)

    # 3) Hough 圆检测
    min_dist = max(1, int(roi_w * min_dist_ratio))
    r_min = int(roi_w * radius_ratio[0])
    r_max = int(roi_w * radius_ratio[1])
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=canny_thresh2,
        param2=hough_param2,
        minRadius=r_min,
        maxRadius=r_max,
    )

    # 4) 没检测到任何圆
    if circles is None:
        return np.zeros((h, w), np.uint8)

    # 5) 取最左和最右各一个圆
    circles = np.round(circles[0]).astype(int)
    # 加上 ROI y_offset
    circles[:, 1] += y_min

    # 按 cx 排序，左右各一个
    circles = sorted(circles, key=lambda c: c[0])
    chosen = []
    if len(circles) >= 1:
        chosen.append(circles[0])
    if len(circles) >= 2:
        chosen.append(circles[-1])

    # 6) 画圆掩膜
    eyes_mask = np.zeros((h, w), np.uint8)
    for x, y, r in chosen:
        cv2.circle(eyes_mask, (x, y), r, 255, thickness=-1)

    # 7) 可选羽化
    kernel = int(r / 2) * 2 + 1
    eyes_mask = cv2.GaussianBlur(eyes_mask, (kernel, kernel), 0)
    _, eyes_mask = cv2.threshold(eyes_mask, 128, 255, cv2.THRESH_BINARY)

    return eyes_mask
