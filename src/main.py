import cv2
import matplotlib.pyplot as plt
import numpy as np

from beautify import get_eyes_mask, get_eyes_mask_hough, slim_face, whiten_lab_gain
from calibrate import undistort_image
from detect_face import get_mask
from preprocess import gaussian_filter, laplacian_sharpen, median_filter, unsharp_mask

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def main():
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    print(f"camera_matrix: {K}")
    print(f"dist_coeffs: {d}")

    plt.figure(figsize=(8, 6))

    img0 = cv2.imread("images/111.jpg")
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("原图")

    img1 = undistort_image(img0, K, d)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("去畸变")

    plt.tight_layout()
    plt.show()

    # 预处理，中值滤波+高斯滤波+拉普拉斯锐化
    plt.figure(figsize=(12, 6))

    img2 = median_filter(img1, k_size=5)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("中值滤波")

    img3 = gaussian_filter(img2, k_size=13, sigma=2.0)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("高斯滤波")

    img4 = laplacian_sharpen(img3, alpha=0.8, k_size=1)
    img41 = unsharp_mask(img3, k_size=(5, 5), sigma=1.0, amount=1.5)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("拉普拉斯锐化")

    plt.tight_layout()
    plt.show()

    # 分割人脸区域，得到人脸掩膜
    plt.figure(figsize=(8, 6))

    face_mask, eyes_mask = get_mask(img4)
    plt.subplot(1, 3, 1)
    plt.imshow(face_mask, cmap="gray")
    plt.axis("off")
    plt.title("人脸掩膜")

    eyes_mask1 = get_eyes_mask(img4)
    eyes_mask2 = get_eyes_mask_hough(
        img4,
        face_mask,
        dp=1,
        min_dist_ratio=50,
        canny_thresh1=30,
        canny_thresh2=100,
        hough_param2=15,
        radius_ratio=(0.05, 0.15),
    )
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

    # 美白
    plt.figure(figsize=(8, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("美白前")

    img5 = whiten_lab_gain(
        img4, face_mask, gain_l=1.25, gain_a=1.0, gain_b=1.0, k_size=101
    )
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("美白后")

    plt.tight_layout()
    # plt.show()

    # 瘦脸
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("瘦脸前")

    img6 = slim_face(img5, face_mask, gain=0.9, k_size=25)
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("瘦脸后")

    plt.tight_layout()
    # plt.show()

    # 大眼

    # 磨皮


if __name__ == "__main__":
    main()
