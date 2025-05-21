import cv2
import matplotlib.pyplot as plt
import numpy as np

from beautify import enlarge_eyes, smooth_skin_guided, squeeze_face, whiten_lab_clahe
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
    img3 = gaussian_filter(img2, k_size=15, sigma=2.0)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

    # 预处理，拉普拉斯锐化
    img4 = laplacian_sharpen(img3, alpha=1.0, k_size=1)
    plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))

    # 分割人脸区域，得到人脸掩膜
    face_mask = get_face_mask(img4)
    plt.imshow(face_mask, cmap="gray")
    plt.show()

    # 4. 美白（CLAHE）
    img5 = whiten_lab_clahe(
        img4, face_mask, clip_limit=2.0, tile_grid_size=(200, 200), k_size=21
    )
    plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    # plt.show()

    img6 = smooth_skin_guided(img5, face_mask, radius=8, eps=1e-2, k_size=11)
    cv2.imwrite("images/skin_guided.jpg", img6)

    # 6. 瘦脸（水平压缩）
    img6 = squeeze_face(img5, face_mask, gain=0.9)
    plt.imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    # plt.show()

    # 7. 大眼（自动检测 + 径向放大）
    img7 = enlarge_eyes(img6, face_mask, gain=1.1)

    # 5. 磨皮（双边滤波）
    img8 = smooth_skin_guided(img7, face_mask)

    # 8. 融合：羽化掩膜 + 加权叠加
    alpha = feather_mask(face_mask, radius=15)
    result = blend_image(img8, img, alpha)


if __name__ == "__main__":
    main()
