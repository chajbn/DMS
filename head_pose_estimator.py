"""Head pose estimation using 3D-2D point correspondences and solvePnP.

Also supports extracting head pose from MediaPipe's facial transformation matrix.
"""

import cv2
import numpy as np


class HeadPoseEstimator:
    """Estimates head rotation (yaw, pitch, roll) from facial landmarks."""

    # MediaPipe face mesh indices for key landmarks
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    DEFAULT_LANDMARK_INDICES = [
        NOSE_TIP,
        CHIN,
        LEFT_EYE_OUTER,
        RIGHT_EYE_OUTER,
        LEFT_MOUTH_CORNER,
        RIGHT_MOUTH_CORNER,
    ]

    # Approximate 3D model points (in mm) for an average adult face
    MODEL_POINTS = np.array([
        [0.0, 0.0, 0.0],         # nose tip
        [0.0, -63.6, -12.5],     # chin
        [-65.6, 23.6, -28.4],    # left eye outer corner
        [65.6, 23.6, -28.4],     # right eye outer corner
        [-61.1, -43.4, -18.9],   # left mouth corner
        [61.1, -43.4, -18.9],    # right mouth corner
    ], dtype=np.float32)

    def __init__(self, landmark_indices=None, model_points=None):
        self.landmark_indices = landmark_indices or self.DEFAULT_LANDMARK_INDICES
        self.model_points = model_points or self.MODEL_POINTS
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def _build_camera_matrix(self, image_w, image_h):
        focal_length = image_w
        cx, cy = image_w / 2.0, image_h / 2.0
        self._camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1],
        ], dtype=np.float32)

    def _from_transform_matrix(self, transform):
        """Extract Euler angles from 4x4 facial transformation matrix."""
        if transform is None:
            return None
        rotation_mat = transform[:3, :3]
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
            roll = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
        else:
            pitch = np.arctan2(-rotation_mat[2, 0], sy)
            yaw = np.arctan2(-rotation_mat[0, 1], rotation_mat[1, 1])
            roll = 0.0
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def _rotation_to_euler(self, R):
        """Convert rotation matrix to Euler angles (yaw, pitch, roll) in degrees.

        R transforms model coords to camera coords.
        Model: +x→right, +y→up, +z→back (nose tip at origin)
        Camera: +x→right, +y→down, +z→forward (into scene)

        Returns (yaw, pitch, roll) where:
          yaw:  head turning left (-) / right (+) — rotation around model Y axis
          pitch: head looking down (-) / up (+)  — rotation around model X axis
          roll:  head tilting left (-) / right (+) — rotation around model Z axis
        """
        # Flip Y and Z axes to convert model coords to standard camera convention
        # Model y(up)→-y, Model z(back)→-z aligns with camera (y↓, z→)
        F = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        R_adj = F @ R @ F.T

        sy = np.sqrt(R_adj[0, 0] ** 2 + R_adj[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-R_adj[2, 0], sy)
            yaw = np.arctan2(R_adj[1, 0], R_adj[0, 0])
            roll = np.arctan2(R_adj[2, 1], R_adj[2, 2])
        else:
            pitch = np.arctan2(-R_adj[2, 0], sy)
            yaw = np.arctan2(-R_adj[0, 1], R_adj[1, 1])
            roll = 0.0

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def estimate(self, landmarks_pixel, image_w, image_h, transform=None):
        """Estimate head pose from pixel landmarks.

        Args:
            landmarks_pixel: Nx2 array of pixel coordinates
            image_w, image_h: image dimensions
            transform: optional 4x4 facial transformation matrix from MediaPipe

        Returns:
            (yaw, pitch, roll) in degrees, or None if estimation fails.
            yaw: left (-) / right (+)
            pitch: down (-) / up (+)
            roll: tilt
        """
        if len(landmarks_pixel) == 0:
            return None

        # Prefer MediaPipe's facial transformation matrix when available
        if transform is not None and transform.shape == (4, 4):
            R = transform[:3, :3]
            return self._rotation_to_euler(R)

        if self._camera_matrix is None:
            self._build_camera_matrix(image_w, image_h)

        image_points = np.array(
            [landmarks_pixel[i] for i in self.landmark_indices],
            dtype=np.float32,
        )

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        return self._rotation_to_euler(rotation_mat)

    def draw_axes(self, image, landmarks_pixel, image_w, image_h, axis_length=50):
        """Draw a 3D axis on the nose tip to visualize head pose."""
        if len(landmarks_pixel) == 0:
            return image

        if self._camera_matrix is None:
            self._build_camera_matrix(image_w, image_h)

        image_points = np.array(
            [landmarks_pixel[i] for i in self.landmark_indices],
            dtype=np.float32,
        )

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return image

        nose_3d = self.model_points[0]  # nose tip
        axis_3d = np.array([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ], dtype=np.float32)

        axis_points_3d = nose_3d + axis_3d
        nose_2d = landmarks_pixel[self.NOSE_TIP]

        axis_2d, _ = cv2.projectPoints(
            axis_points_3d, rotation_vec, translation_vec,
            self._camera_matrix, self._dist_coeffs,
        )

        nose_pt = tuple(map(int, nose_2d))
        cv2.line(image, nose_pt, tuple(map(int, axis_2d[0][0])), (0, 0, 255), 2)  # X: red
        cv2.line(image, nose_pt, tuple(map(int, axis_2d[1][0])), (0, 255, 0), 2)  # Y: green
        cv2.line(image, nose_pt, tuple(map(int, axis_2d[2][0])), (255, 0, 0), 2)  # Z: blue

        return image
