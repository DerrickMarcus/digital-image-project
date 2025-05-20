import cv2
import matplotlib.pyplot as plt
import numpy as np

from beautify import enlarge_eyes, smooth_skin, squeeze_face, whiten_face
from blend import blend_image, feather_mask
from calibrate import undistort_image
from detect_face import get_face_mask
from preprocess import gaussian_filter, laplacian_sharpen, median_filter


def main():
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    # print(f"camera_matrix: {K}")
    # print(f"dist_coeffs: {d}")

    img = cv2.imread("images/111.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 去畸变
    img1 = undistort_image(img, K, d)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    # 预处理，中值滤波+高斯滤波
    img2 = median_filter(img1, k_size=5)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img3 = gaussian_filter(img2, k_size=5, sigma=1.0)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

    # 预处理，拉普拉斯锐化
    img4 = laplacian_sharpen(img3, alpha=1.5)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # 分割人脸区域，得到人脸掩膜
    face_mask = get_face_mask(img2)
    mask = face_mask.astype(bool)
    img3 = np.zeros_like(img2)
    img3[mask] = img2[mask]

    # 4. 美白（CLAHE）
    white = whiten_face(img3, mask, clip_limit=2.0, tile_grid_size=(8, 8))

    # 5. 磨皮（双边滤波）
    smooth = smooth_skin(white, mask, diameter=9, sigma_color=75, sigma_space=75)

    # 6. 瘦脸（水平压缩）
    slim = squeeze_face(smooth, mask, strength=0.9)

    # 7. 大眼（自动检测 + 径向放大）
    big = enlarge_eyes(slim, mask, gain=1.1)

    # 8. 融合：羽化掩膜 + 加权叠加
    alpha = feather_mask(mask, radius=15)
    result = blend_image(big, img, alpha)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
