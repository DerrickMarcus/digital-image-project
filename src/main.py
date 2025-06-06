from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from beautify import enlarge_eyes, slim_face, smooth_skin_guided, whiten_lab_gain
from calibrate import undistort_image
from detect_face import get_face_mask
from preprocess import clahe_equalization, unsharp_mask

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def process(image_path: Path):
    params = np.load("src/calib_params.npz")
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    # print(f"camera_matrix: {K}")
    # print(f"dist_coeffs: {d}")

    # plt.figure(figsize=(8, 6))

    # 读取图像，去畸变
    img0 = cv2.imread(str(image_path))
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

    img2 = cv2.medianBlur(img1, ksize=5)
    # plt.subplot(1, 4, 1)
    # plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("中值滤波")

    img3 = cv2.GaussianBlur(img2, (13, 13), 0)
    # plt.subplot(1, 4, 2)
    # plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("高斯滤波")

    img4 = unsharp_mask(img3, k_size=(5, 5), sigma=1.0, amount=1.5)
    # plt.subplot(1, 4, 3)
    # plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("高频提升 反锐化掩膜")

    img5 = clahe_equalization(img4, clip_limit=2.0, tile_grid_size=(200, 200))
    # plt.subplot(1, 4, 4)
    # plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("CLAHE 均衡化")

    # plt.tight_layout()
    # plt.show()

    # 分割人脸区域，得到人脸掩膜
    # plt.figure(figsize=(8, 6))

    face_mask = get_face_mask(img5)
    # plt.subplot(1, 2, 1)
    # plt.imshow(face_mask, cmap="gray")
    # plt.axis("off")
    # plt.title("人脸掩膜")

    face = cv2.bitwise_and(img5, img5, mask=face_mask)
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("人脸区域")

    # plt.tight_layout()
    # plt.show()

    # 美白
    # plt.figure(figsize=(8, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("美白前")

    img6 = whiten_lab_gain(
        img5, face_mask, gain_l=1.10, gain_a=1.0, gain_b=1.0, k_size=51
    )
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("美白后")

    # plt.tight_layout()
    # plt.show()

    # 瘦脸
    # plt.figure(figsize=(8, 6))

    # plt.subplot(2, 1, 1)
    # plt.imshow(cv2.cvtColor(img6, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("瘦脸前")

    img7 = slim_face(img6, face_mask, gain=0.94, k_size=51)
    face_mask = slim_face(face_mask, face_mask, gain=0.94, k_size=51)
    _, face_mask = cv2.threshold(face_mask, 128, 255, cv2.THRESH_BINARY)
    # plt.subplot(2, 1, 2)
    # plt.imshow(cv2.cvtColor(img7, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("瘦脸后")

    # plt.tight_layout()
    # plt.show()

    # 大眼
    # plt.figure(figsize=(8, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img7, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("大眼前")

    img8 = enlarge_eyes(
        img7, face_mask, gain=1.2, distance=(0.12, 0.10, 0.12, 0.20, 0.60), k_size=101
    )
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img8, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("大眼后")

    # plt.tight_layout()
    # plt.show()

    # 磨皮
    # plt.figure(figsize=(8, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img8, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("磨皮前")

    img9 = smooth_skin_guided(img8, face_mask, radius=6, eps=0.01, k_size=11)
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img9, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("磨皮后")

    # plt.tight_layout()
    # plt.show()

    # 与原图融合
    # plt.figure(figsize=(8, 6))

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("原图")

    img10 = slim_face(img5, face_mask, gain=0.94, k_size=51)
    alpha = cv2.GaussianBlur((face_mask / 255.0).astype(np.float32), (31, 31), 0)[
        ..., None
    ]
    img11 = img10.astype(np.float32) * (1 - alpha) + img9.astype(np.float32) * alpha
    img11 = np.clip(img11, 0, 255).astype(np.uint8)
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img11, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("融合后")

    # plt.tight_layout()
    # plt.show()

    sub_path = image_path.parent / "processed"
    sub_path.mkdir(parents=True, exist_ok=True)
    new_path = sub_path / (image_path.stem + "_new" + image_path.suffix)
    cv2.imwrite(str(new_path), img11)
    print(f"Processed {image_path.name} and saved as {new_path.name}")


if __name__ == "__main__":
    images_dir = Path("images")
    for image in images_dir.glob("*.jpg"):
        process(image)
