import cv2
import matplotlib.pyplot as plt
import numpy as np

# from beautify import enlarge_eyes, smooth_skin, squeeze_face, whiten_face
# from blend import blend_image, feather_mask
from detect_face import adaptive_threshold, morphological
from preprocess import gaussian_filter


def main():
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    print(f"camera_matrix: {K}")
    print(f"dist_coeffs: {d}")

    img = cv2.imread("images/111.jpg")

    # 2. 简单预处理（高斯去噪）
    pre = gaussian_filter(img, k_size=5, sigma=1.0)

    # # 3. 掩膜分割：Otsu + 开闭运算清理
    gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    mask = adaptive_threshold(gray)
    mask = morphological(mask, op="open", k_size=5)
    mask = morphological(mask, op="close", k_size=5)

    # # 4. 美白（CLAHE）
    # white = whiten_face(pre, mask, clip_limit=2.0, tile_grid_size=(8, 8))

    # # 5. 磨皮（双边滤波）
    # smooth = smooth_skin(white, mask, diameter=9, sigma_color=75, sigma_space=75)

    # # 6. 瘦脸（水平压缩）
    # slim = squeeze_face(smooth, mask, strength=0.9)

    # # 7. 大眼（自动检测 + 径向放大）
    # big = enlarge_eyes(slim, mask, gain=1.1)

    # # 8. 融合：羽化掩膜 + 加权叠加
    # alpha = feather_mask(mask, radius=15)
    # result = blend_image(big, img, alpha)

    plt.imshow(mask, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
