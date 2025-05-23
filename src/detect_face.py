import cv2
import matplotlib.pyplot as plt
import numpy as np

from calibrate import undistort_image
from preprocess import laplacian_sharpen


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


if __name__ == "__main__":
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    print(f"camera_matrix: {K}")
    print(f"dist_coeffs: {d}")

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    img0 = cv2.imread("images/222.jpg")
    img1 = undistort_image(img0, K, d)
    img2 = cv2.medianBlur(img1, ksize=5)
    img3 = cv2.GaussianBlur(img2, (13, 13))
    img4 = laplacian_sharpen(img3, alpha=0.8, k_size=1)

    # 分割人脸区域，得到人脸掩膜
    plt.figure(figsize=(8, 6))

    face_mask = get_face_mask(img4)
    plt.subplot(1, 2, 1)
    plt.imshow(face_mask, cmap="gray")
    plt.axis("off")
    plt.title("人脸掩膜")

    face = cv2.bitwise_and(img4, img4, mask=face_mask)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("人脸区域")

    plt.tight_layout()
    plt.show()
