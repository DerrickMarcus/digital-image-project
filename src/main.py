import cv2
import matplotlib.pyplot as plt
import numpy as np

from beautify import (
    whiten_hsi_clahe,
    whiten_hsi_gain,
    whiten_hsi_gamma,
    whiten_lab_clahe,
    whiten_lab_gain,
)
from calibrate import undistort_image
from detect_face import get_face_mask
from preprocess import gaussian_filter, laplacian_sharpen, median_filter

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def main():
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    print(f"camera_matrix: {K}")
    print(f"dist_coeffs: {d}")

    img = cv2.imread("images/111.jpg")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.figure(figsize=(10, 8))
    # 去畸变
    img1 = undistort_image(img, K, d)
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("去畸变")

    # 预处理，中值滤波+高斯滤波
    img2 = median_filter(img1, k_size=5)
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("中值滤波")
    img3 = gaussian_filter(img2, k_size=15, sigma=2.0)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("高斯滤波")

    # 预处理，拉普拉斯锐化
    img4 = laplacian_sharpen(img3, alpha=1.0, k_size=1)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("拉普拉斯锐化")

    plt.tight_layout()
    plt.show()

    # 分割人脸区域，得到人脸掩膜
    plt.figure()
    face_mask = get_face_mask(img4)
    plt.imshow(face_mask, cmap="gray")
    plt.axis("off")
    plt.show()

    # 美白
    plt.figure(figsize=(15, 10))
    img51 = whiten_hsi_clahe(
        img4, face_mask, clip_limit=2, tile_grid_size=(200, 200), k_size=21
    )
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img51, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("hsi, clahe")

    img52 = whiten_lab_clahe(
        img4, face_mask, clip_limit=2, tile_grid_size=(200, 200), k_size=21
    )
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img52, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("lab, clahe")

    img53 = whiten_hsi_gamma(img4, face_mask, gamma=0.7, k_size=21)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(img53, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("hsi, gamma")

    img54 = whiten_hsi_gain(img4, face_mask, gain_i=1.2, gain_s=1.0, k_size=21)
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img54, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("hsi, gain")

    img55 = whiten_lab_gain(
        img4, face_mask, gain_l=1.2, gain_a=1.0, gain_b=1.0, k_size=21
    )
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img55, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("lab, gain")

    plt.tight_layout()
    plt.show()

    # 瘦脸

    # 大眼

    # 磨皮


if __name__ == "__main__":
    main()
