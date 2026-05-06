"""Face mesh detection using MediaPipe Face Landmarker (Task API)."""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


_MODEL_NAME = 'face_landmarker.task'


class FaceMeshDetector:
    """Detects 478 facial landmarks using MediaPipe Face Landmarker.

    Also provides blendshapes (52 coefficients) and facial transformation matrix.
    """

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5,
                 model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), _MODEL_NAME)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, image_bgr):
        """Detect face landmarks in a BGR image.

        Returns:
            dict with:
              - landmarks_pixel: Nx2 array of pixel coordinates
              - landmarks_3d: Nx3 array of normalized (x,y,z) coords
              - blendshapes: dict mapping Blendshape name -> score, or None
              - transform: 4x4 facial transformation matrix, or None
            If no face found, returns dict with empty arrays and Nones.
        """
        h, w = image_bgr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_bgr)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {
                'landmarks_pixel': np.array([], dtype=np.int32).reshape(0, 2),
                'landmarks_3d': np.array([], dtype=np.float32).reshape(0, 3),
                'blendshapes': None,
                'transform': None,
            }

        face_lms = result.face_landmarks[0]

        # Convert to pixel and 3d coordinates
        landmarks_pixel = np.array([
            [int(lm.x * w), int(lm.y * h)] for lm in face_lms
        ], dtype=np.int32)
        landmarks_3d = np.array([
            [lm.x, lm.y, lm.z] for lm in face_lms
        ], dtype=np.float32)

        # Extract blendshapes
        blendshapes = None
        if result.face_blendshapes:
            blendshapes = {}
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score

        # Extract transformation matrix
        transform = None
        if result.facial_transformation_matrixes:
            transform = result.facial_transformation_matrixes[0]

        return {
            'landmarks_pixel': landmarks_pixel,
            'landmarks_3d': landmarks_3d,
            'blendshapes': blendshapes,
            'transform': transform,
        }

    def draw_landmarks(self, image, landmarks_pixel, indices=None):
        """Draw selected landmarks on the image."""
        if len(landmarks_pixel) == 0:
            return image

        if indices is None:
            indices = range(len(landmarks_pixel))

        for i in indices:
            x, y = landmarks_pixel[i]
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

        return image

    def release(self):
        self.landmarker.close()
