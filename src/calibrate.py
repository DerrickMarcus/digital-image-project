import glob

import cv2
import numpy as np


def detect_corners(
    images_path: str, board_size: tuple[int, int], square_size: float
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Detect chessboard corners in calibration images and prepare object/image points.

    Args:
        images_path (str): Glob pattern to calibration images.
        board_size (tuple[int, int]): Number of corners in chessboard (cols, rows).
        square_size (float): Size of one square edge on the board.

    Returns:
        object_points (list[np.ndarray]): 3D points in real world space for each image.
        image_points (list[np.ndarray]): 2D points in image plane for each image.
        image_shape (tuple[int, int]): Shape of calibration images (width, height).
    """
    obj_p = np.zeros((board_size[1] * board_size[0], 3), np.float32)
    obj_p[:, :2] = np.indices((board_size[0], board_size[1])).T.reshape(-1, 2)
    obj_p *= square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_shape: tuple[int, int] = (0, 0)

    for fname in glob.glob(images_path):
        img = cv2.imread(fname)
        if img is None:
            continue
        if image_shape == (0, 0):
            image_shape = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"Warning: Chessboard not found in {fname}")
            continue

        cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        object_points.append(obj_p)
        image_points.append(corners)

        cv2.drawChessboardCorners(img, board_size, corners, found)
        cv2.namedWindow("Detected Corners", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(2000)

    cv2.destroyAllWindows()

    return object_points, image_points, image_shape


def calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Perform camera calibration to compute intrinsic matrix and distortion coefficients.

    Args:
        object_points (list[np.ndarray]): 3D world points from detect_corners().
        image_points (list[np.ndarray]): 2D image points from detect_corners().
        image_shape (tuple[int, int]): (width, height) of calibration images.

    Returns:
        camera_matrix (np.ndarray): Intrinsic parameters matrix (3x3).
        dist_coeffs (np.ndarray): Distortion coefficients (k1, k2, p1, p2, k3).
    """
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_shape, None, None
    )

    total_error = 0
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    mean_error = total_error / len(object_points)
    print(f"Calibration completed. Mean reprojection error: {mean_error:.4f}px")

    return camera_matrix, dist_coeffs


def undistort_image(
    image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    """Undistort an image using camera matrix and distortion coefficients.

    Args:
        image (np.ndarray): Input distorted image.
        camera_matrix (np.ndarray): Intrinsic matrix from calibrate_camera().
        dist_coeffs (np.ndarray): Distortion coefficients from calibrate_camera().

    Returns:
        np.ndarray: Undistorted image.
    """
    h, w = image.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_mtx)


if __name__ == "__main__":
    pattern = "images/chessboard/*.jpg"
    board_size = (10, 7)
    square_size = 25
    obj_p, img_p, shape = detect_corners(pattern, board_size, square_size)
    K, d = calibrate_camera(obj_p, img_p, shape)
    np.savez("src/calib_params.npz", camera_matrix=K, dist_coeffs=d)
