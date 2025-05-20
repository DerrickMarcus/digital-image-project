import cv2
import numpy as np


def get_face_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    mask = (
        cv2.inRange(hsv, (0, 40, 40), (25, 120, 240))
        | cv2.inRange(hsv, (155, 40, 40), (179, 120, 240))
    ) & cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    cr = ycrcb[:, :, 1]
    _, mask_cr = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask & mask_cr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(final, [c], -1, 255, thickness=-1)

    return mask


def global_threshold(gray: np.ndarray, thresh: int = 128) -> np.ndarray:
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(gray: np.ndarray) -> tuple[np.ndarray, int]:
    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, int(thresh_val)


def adaptive_threshold(
    gray: np.ndarray, max_value: int = 255, block_size: int = 11, C: int = 2
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


def morphological(mask: np.ndarray, op: str = "open", k_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    if op == "erode":
        return cv2.erode(mask, kernel)
    elif op == "dilate":
        return cv2.dilate(mask, kernel)
    elif op == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif op == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif op == "tophat":
        return cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
    elif op == "blackhat":
        return cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise ValueError(f"Unsupported operation: {op}.")
