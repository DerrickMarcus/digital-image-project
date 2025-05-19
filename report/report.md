# 数字图像处理-课程大作业-人脸美颜

> 姓名：陈彦旭
>
> 班级：无24

## 整体流程

### 相机畸变矫正

模块 `calibrate.py` ：

`detect_corners()` 函数，加载棋盘格图片，检测和细化图像中角点，返回各个图像中角点在真实世界坐标系中的位置，和在图像坐标系中的位置。

`calibrate_camera()` 函数，使用 `detect_corners()` 函数返回的角点位置，计算相机内参矩阵和畸变系数.

`undistort_image()` 函数，接受一张存在畸变的图像，根据前面计算出的内参矩阵和畸变系数，返回去畸变的图像。

### 图像预处理

模块 `preprocess.py` ：

色彩空间转换：

1. `rgb_to_hsi()` 函数，将 RGB 图像转换为 HSI 图像。
2. `hsi_to_rgb()` 函数，将 HSI 图像转换为 RGB 图像。

3种空域滤波器：

1. 中值滤波器 `median_filter()` 函数。
2. 高斯滤波器 `gaussian_filter()` 函数。
3. 双边滤波器 `bilateral_filter()` 函数。

2种直方图均衡化：

1. `histogram_equalization()` 函数，全局均衡化。
2. `clahe_equalization()` 函数，局部自适应均衡化。

2种图像高频提升：

1. `laplacian_sharpen()` 函数，拉普拉斯锐化。
2. `unsharp_mask()` 函数，反锐化掩模。

### 人脸检测与分割

模块 `detect_face.py` ：

3种阈值分割，分割出人脸区域，得到二值化图像：

1. `global_threshold()` 函数，全局阈值分割。
2. `otsu_threshold()` 函数，大津算法阈值分割。
3. `adaptive_threshold()` 函数，自适应阈值分割。

2两种边缘检测：

1. `soble_edge()` 函数，Soble 算子边缘检测。
2. `canny_edge()` 函数，Canny 算子边缘检测。

`morphological` 函数，对二值掩膜进行形态学操作，支持腐蚀、膨胀、开运算、闭运算、顶帽、黑帽等操作。

### 人脸美艳

`beautigy.py` 模块：

`whiten_face()` 美白函数，将图片转换到 HSI，仅对亮度通道进行 CLAHE 均衡化，再与面部掩膜融合，最后转换回 RGB。

`smooth_skin()` 磨皮函数，对图像进行双边滤波处理，再与面部掩膜融合。

`squeeze_face()` 瘦脸函数，对面部区域做仿射水平压缩（在 X 方向上缩放因子 < 1），再与原图融合。

`enlarge_eye()` 大眼函数，在面部上半区域。用 Canny 边缘检测和形态学操作找到两个眼睛亮区的连通域，得到眼睛中心的坐标和眼睛的大小，使用局部的径向放缩函数，再使用掩膜融合到原图。

### 图像后处理

`blend.py` 模块：在之前对人脸区域的操作过程中，可能会对图片背景部分造成影响，因此对美颜后图片和原始图像进行融合，且使用渐变融合，使得拼接区域过渡自然。

`feather_mask()` 函数，对人脸区域的二值化掩膜进行高斯模糊，生成羽化掩膜。

`blend_image()` 函数，使用羽化掩膜对美颜后的图像和原始图像进行融合，保证边缘过渡自然。
