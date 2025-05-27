import cv2
import numpy as np


def get_face_mask(image: np.ndarray) -> np.ndarray:
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(face, [c], -1, 255, thickness=-1)

    # mask 未联通，final 已联通
    # 膨胀操作使得轮廓扩大
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    face = cv2.dilate(face, kernel, iterations=1)

    face = cv2.GaussianBlur(face, (51, 51), 0)
    _, face = cv2.threshold(face, 128, 255, cv2.THRESH_BINARY)

    return face


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
