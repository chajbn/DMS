"""Mouth closure detection using facial landmarks.

Computes Mouth Aspect Ratio (MAR) from MediaPipe face mesh landmarks
to determine mouth openness / closure.
"""

import cv2
import numpy as np


class MouthDetector:
    """Detects mouth openness based on facial landmarks.

    Uses the Mouth Aspect Ratio (MAR) — the ratio of inner lip vertical
    distance to mouth corner horizontal distance.
      MAR = ||upper_lip - lower_lip|| / ||left_corner - right_corner||

    Low MAR  → mouth closed
    Mid MAR  → talking
    High MAR → yawning
    """

    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP_INNER = 13
    LOWER_LIP_INNER = 14

    # Outer lip landmarks (alternative / supplemental)
    UPPER_LIP_TOP = 0
    LOWER_LIP_BOTTOM = 17

    def __init__(self, mar_yawn_threshold=0.65, mar_open_threshold=0.35):
        self.mar_yawn_threshold = mar_yawn_threshold
        self.mar_open_threshold = mar_open_threshold

    def compute_mar(self, landmarks_pixel):
        """Compute Mouth Aspect Ratio from facial landmarks.

        Args:
            landmarks_pixel: Nx2 array of pixel coordinates

        Returns:
            float: MAR value, or None if landmarks unavailable
        """
        if len(landmarks_pixel) == 0:
            return None

        try:
            upper = np.array(landmarks_pixel[self.UPPER_LIP_INNER], dtype=np.float32)
            lower = np.array(landmarks_pixel[self.LOWER_LIP_INNER], dtype=np.float32)
            left = np.array(landmarks_pixel[self.MOUTH_LEFT], dtype=np.float32)
            right = np.array(landmarks_pixel[self.MOUTH_RIGHT], dtype=np.float32)

            vertical = np.linalg.norm(upper - lower)
            horizontal = np.linalg.norm(left - right)

            if horizontal < 1e-6:
                return 0.0

            return float(vertical / horizontal)
        except (IndexError, KeyError):
            return None

    def is_open(self, mar):
        """Check if mouth is open beyond the talking threshold."""
        if mar is None:
            return False
        return mar > self.mar_open_threshold

    def is_yawning(self, mar):
        """Check if mouth is wide enough to indicate yawning."""
        if mar is None:
            return False
        return mar > self.mar_yawn_threshold

    def draw(self, image, landmarks_pixel):
        """Highlight mouth landmarks used for MAR computation."""
        if len(landmarks_pixel) == 0:
            return image

        try:
            upper = tuple(landmarks_pixel[self.UPPER_LIP_INNER])
            lower = tuple(landmarks_pixel[self.LOWER_LIP_INNER])
            left = tuple(landmarks_pixel[self.MOUTH_LEFT])
            right = tuple(landmarks_pixel[self.MOUTH_RIGHT])

            # Vertical line (inner lip gap)
            cv2.line(image, upper, lower, (0, 255, 255), 1)
            # Horizontal line (mouth width)
            cv2.line(image, left, right, (0, 255, 255), 1)

            # Corner points
            for pt in (upper, lower, left, right):
                cv2.circle(image, pt, 2, (0, 255, 255), -1)
        except (IndexError, KeyError):
            pass

        return image
