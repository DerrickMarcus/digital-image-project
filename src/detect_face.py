import cv2
import matplotlib.pyplot as plt
import numpy as np

from calibrate import undistort_image
from preprocess import gaussian_filter, laplacian_sharpen, median_filter, unsharp_mask


def get_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    mask = (
        cv2.inRange(hsv, (0, 50, 60), (25, 120, 240))
        | cv2.inRange(hsv, (155, 50, 60), (179, 120, 240))
        | cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        # | cv2.inRange(lab[..., 0], 0, 128)
        # | cv2.inRange(lab, (0, 128 - 5, 128 - 5), (255, 128 - 5, 128 + 20))
    )

    cr = ycrcb[:, :, 1]
    _, mask_cr = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask | mask_cr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(face, [c], -1, 255, thickness=-1)

    # mask 未联通，final 已联通
    # eyes = cv2.bitwise_and(face, cv2.bitwise_not(mask))
    eyes = face - mask

    # 膨胀操作使得轮廓扩大
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    face = cv2.dilate(face, kernel, iterations=1)

    face = cv2.GaussianBlur(face, (51, 51), 0)
    _, face = cv2.threshold(face, 128, 255, cv2.THRESH_BINARY)

    # eyes = face - mask
    ys, _ = np.where(face > 0)
    ymid = int((ys.min() + ys.max()) / 2)
    eyes[ymid:, :] = 0

    # 5) 小开去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eyes = cv2.morphologyEx(eyes, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) 连通域取两大块
    cnts, _ = cv2.findContours(eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    eyes2 = np.zeros_like(eyes)
    for c in cnts:
        if cv2.contourArea(c) < 50:
            continue
        if len(c) >= 5:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
            cv2.ellipse(eyes2, ((cx, cy), (MA, ma), ang), 255, -1)
        else:
            cv2.drawContours(eyes2, [c], -1, 255, -1)

    # 7) 最后羽化收紧
    eyes2 = cv2.GaussianBlur(eyes2, (31, 31), 0)
    _, eyes2 = cv2.threshold(eyes2, 128, 255, cv2.THRESH_BINARY)

    return face, eyes2


def global_threshold(gray: np.ndarray, thresh: int = 128) -> np.ndarray:
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(gray: np.ndarray) -> tuple[np.ndarray, int]:
    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, int(thresh_val)


def adaptive_threshold(
    gray: np.ndarray, max_value: int = 255, block_size: int = 301, C: int = 2
) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )
    return binary


def sobel_edge(gray: np.ndarray, k_size: int = 3) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k_size)

    mag = np.sqrt(grad_x**2 + grad_y**2)
    mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
    return mag


def canny_edge(
    gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150
) -> np.ndarray:
    return cv2.Canny(gray, low_thresh, high_thresh)


def enlarge_eyes(
    image: np.ndarray,
    eyes_mask: np.ndarray,
    gain: float = 1.2,
    dilate_ksize: int = 31,
    blur_ksize: int = 51,
) -> np.ndarray:
    """
    Enlarge the eyes in an image by radially scaling pixels within the eye regions.

    Args:
        image:         Input BGR image, shape (H, W, 3).
        eyes_mask:     Binary mask of eye regions (0 or 255), shape (H, W).
        gain:          Radial scale factor (>1 enlarges).
        dilate_ksize:  Kernel size for dilating eyes_mask to include eye whites.
        blur_ksize:    Kernel size for Gaussian blur on the mask (must be odd).

    Returns:
        out:           Output BGR image with eyes enlarged.
    """
    h, w = image.shape[:2]
    # 1) Expand mask to cover full eye area (whites, lashes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    mask_big = cv2.dilate(eyes_mask, kernel, iterations=1)
    # ensure binary
    _, mask_big = cv2.threshold(mask_big, 128, 255, cv2.THRESH_BINARY)

    # 2) Find each eye contour and its enclosing circle
    cnts, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eyes = []
    for c in cnts:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        eyes.append((cx, cy, r))
    if not eyes:
        return image.copy()

    # 3) Build mapping grids
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # 4) Apply radial scaling for each eye
    for cx, cy, r in eyes:
        dx = map_x - cx
        dy = map_y - cy
        mask_circle = (dx * dx + dy * dy) <= (r * r)
        map_x[mask_circle] = cx + dx[mask_circle] * gain
        map_y[mask_circle] = cy + dy[mask_circle] * gain

    # 5) Clip coordinates
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    # 6) Remap image
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # 7) Generate smooth alpha from mask_big
    alpha = mask_big.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)
    alpha = alpha[..., None]  # shape (H, W, 1)

    # 8) Blend warped eyes with original
    out = warped.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    print(f"camera_matrix: {K}")
    print(f"dist_coeffs: {d}")

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    # plt.figure(figsize=(8, 6))

    img0 = cv2.imread("images/111.jpg")
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("原图")

    img1 = undistort_image(img0, K, d)
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("去畸变")

    # plt.tight_layout()
    # plt.show()

    # 预处理，中值滤波+高斯滤波+拉普拉斯锐化
    # plt.figure(figsize=(12, 6))

    img2 = median_filter(img1, k_size=5)
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("中值滤波")

    img3 = gaussian_filter(img2, k_size=13, sigma=2.0)
    # plt.subplot(1, 3, 2)
    # plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("高斯滤波")

    img4 = laplacian_sharpen(img3, alpha=0.8, k_size=1)
    img41 = unsharp_mask(img3, k_size=(5, 5), sigma=1.0, amount=1.5)
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("拉普拉斯锐化")

    # plt.tight_layout()
    # plt.show()

    # 分割人脸区域，得到人脸掩膜
    plt.figure(figsize=(8, 6))

    face_mask, eyes_mask = get_mask(img4)
    plt.subplot(1, 3, 1)
    plt.imshow(face_mask, cmap="gray")
    plt.axis("off")
    plt.title("人脸掩膜")

    plt.subplot(1, 3, 2)
    plt.imshow(eyes_mask, cmap="gray")
    plt.axis("off")
    plt.title("眼睛掩膜")

    face = cv2.bitwise_and(img4, img4, mask=face_mask)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("人脸区域")

    plt.tight_layout()
    plt.show()

    # 大眼

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("大眼前")

    img5 = enlarge_eyes(img4, eyes_mask, gain=0.6, dilate_ksize=31, blur_ksize=51)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("大眼后")
    plt.show()
