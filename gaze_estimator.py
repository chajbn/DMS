"""Gaze direction estimation using MediaPipe blendshapes and iris landmarks."""

import cv2
import numpy as np


class GazeEstimator:
    """Estimates gaze direction from blendshapes and iris landmarks.

    Uses MediaPipe blendshapes (52 facial expression coefficients) which
    directly encode eye movement directions, providing more reliable gaze
    estimation than manual iris tracking alone.
    """

    # MediaPipe iris landmarks
    LEFT_IRIS_CENTER = 473
    RIGHT_IRIS_CENTER = 468

    # Eye corner landmarks
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    def __init__(self, ear_threshold=0.22):
        self.ear_threshold = ear_threshold

    def _estimate_from_blendshapes(self, blendshapes):
        """Estimate gaze from blendshape coefficients (primary method)."""
        if blendshapes is None:
            return None

        # Eye look direction blendshapes (0-1 range)
        look_up = max(
            blendshapes.get('eyeLookUpLeft', 0),
            blendshapes.get('eyeLookUpRight', 0),
        )
        look_down = max(
            blendshapes.get('eyeLookDownLeft', 0),
            blendshapes.get('eyeLookDownRight', 0),
        )
        look_left = max(
            blendshapes.get('eyeLookInLeft', 0),
            blendshapes.get('eyeLookOutRight', 0),
        )
        look_right = max(
            blendshapes.get('eyeLookOutLeft', 0),
            blendshapes.get('eyeLookInRight', 0),
        )

        # Map blendshape scores to approximate angles
        gaze_yaw = (look_right - look_left) * 45.0
        gaze_pitch = (look_down - look_up) * 30.0

        # Eye blink for drowsiness
        blink_left = blendshapes.get('eyeBlinkLeft', 0)
        blink_right = blendshapes.get('eyeBlinkRight', 0)
        avg_blink = (blink_left + blink_right) / 2.0

        # Convert blink to EAR approximation (inverse relationship)
        # High blink score = closed eye = low EAR
        avg_ear = max(0.0, 1.0 - avg_blink) * 0.35

        return {
            'gaze_yaw': np.clip(gaze_yaw, -45.0, 45.0),
            'gaze_pitch': np.clip(gaze_pitch, -30.0, 30.0),
            'avg_ear': avg_ear,
            'blink_left': blink_left,
            'blink_right': blink_right,
        }

    def _estimate_from_iris(self, landmarks_pixel, image_w, image_h):
        """Estimate gaze from iris position relative to eye corners (fallback)."""
        if len(landmarks_pixel) == 0:
            return None

        try:
            # Left eye
            left_iris = np.array(landmarks_pixel[self.LEFT_IRIS_CENTER], dtype=np.float32)
            left_outer = np.array(landmarks_pixel[self.LEFT_EYE_OUTER], dtype=np.float32)
            left_inner = np.array(landmarks_pixel[self.LEFT_EYE_INNER], dtype=np.float32)
            left_eye_width = np.linalg.norm(left_outer - left_inner)
            left_eye_center = (left_outer + left_inner) / 2.0
            left_gaze = (left_iris - left_eye_center) / max(left_eye_width, 1e-6)

            # Right eye
            right_iris = np.array(landmarks_pixel[self.RIGHT_IRIS_CENTER], dtype=np.float32)
            right_outer = np.array(landmarks_pixel[self.RIGHT_EYE_OUTER], dtype=np.float32)
            right_inner = np.array(landmarks_pixel[self.RIGHT_EYE_INNER], dtype=np.float32)
            right_eye_width = np.linalg.norm(right_outer - right_inner)
            right_eye_center = (right_outer + right_inner) / 2.0
            right_gaze = (right_iris - right_eye_center) / max(right_eye_width, 1e-6)

            avg_gaze_x = (left_gaze[0] + right_gaze[0]) / 2.0
            avg_gaze_y = (left_gaze[1] + right_gaze[1]) / 2.0

            return {
                'gaze_yaw': np.clip(avg_gaze_x * 30.0, -45.0, 45.0),
                'gaze_pitch': np.clip(avg_gaze_y * 20.0, -30.0, 30.0),
                'avg_ear': 0.3,  # default
                'blink_left': 0.0,
                'blink_right': 0.0,
                'left_gaze': tuple(left_gaze),
                'right_gaze': tuple(right_gaze),
            }
        except (IndexError, KeyError):
            return None

    def estimate(self, face_data):
        """Estimate gaze direction.

        Args:
            face_data: dict from FaceMeshDetector.detect() with keys:
              'landmarks_pixel', 'landmarks_3d', 'blendshapes', 'transform'

        Returns:
            dict with gaze_yaw, gaze_pitch, avg_ear, blink_left, blink_right
        """
        blendshapes = face_data.get('blendshapes')
        landmarks_pixel = face_data.get('landmarks_pixel', [])

        # Primary: use blendshapes
        if blendshapes is not None:
            result = self._estimate_from_blendshapes(blendshapes)
            if result is not None:
                return result

        # Fallback: use iris positions
        if len(landmarks_pixel) > 0:
            return self._estimate_from_iris(landmarks_pixel, 0, 0)

        return None

    def draw_gaze(self, image, landmarks_pixel, gaze_result):
        """Draw gaze visualization on the image."""
        if gaze_result is None or len(landmarks_pixel) == 0:
            return image

        h, w = image.shape[:2]

        # Draw iris centers if available (fallback path)
        try:
            left_iris = tuple(landmarks_pixel[self.LEFT_IRIS_CENTER])
            right_iris = tuple(landmarks_pixel[self.RIGHT_IRIS_CENTER])
            cv2.circle(image, left_iris, 2, (255, 255, 0), -1)
            cv2.circle(image, right_iris, 2, (255, 255, 0), -1)
        except (IndexError, KeyError):
            pass

        # Draw gaze direction from nose area
        nose_tip = tuple(landmarks_pixel[1].astype(int)) if len(landmarks_pixel) > 1 else None
        if nose_tip:
            scale = w * 0.15
            gaze_yaw_rad = np.radians(gaze_result.get('gaze_yaw', 0))
            gaze_pitch_rad = np.radians(gaze_result.get('gaze_pitch', 0))
            end_pt = (
                int(nose_tip[0] + np.sin(gaze_yaw_rad) * scale),
                int(nose_tip[1] - np.sin(gaze_pitch_rad) * scale),
            )
            cv2.arrowedLine(image, nose_tip, end_pt, (0, 255, 255), 2, tipLength=0.3)

        return image
