"""DMS (Driver Monitoring System) main pipeline.

Detects facial landmarks -> estimates head pose & gaze -> judges distraction.
Uses MediaPipe Face Landmarker (Task API).
"""

import sys
import time
import cv2
import numpy as np

from face_mesh_detector import FaceMeshDetector
from head_pose_estimator import HeadPoseEstimator
from gaze_estimator import GazeEstimator
from mouth_detector import MouthDetector
from distraction_detector import DistractionDetector, DistractionState


def draw_status_panel(image, details):
    """Draw status information panel on the image."""
    h, w = image.shape[:2]

    # Semi-transparent panel background
    overlay = image.copy()
    panel_h, panel_w = 340, 320
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (30, 30, 30), -1)
    image = cv2.addWeighted(image, 1.0, overlay, 0.5, 0)

    y = 30
    state = details.get('state', 'Unknown')
    state_colors = {
        DistractionState.NORMAL.value: (0, 255, 0),
        DistractionState.SLIGHT.value: (0, 215, 255),
        DistractionState.DISTRACTED.value: (0, 0, 255),
        DistractionState.DROWSY.value: (255, 0, 255),
        DistractionState.NO_FACE.value: (128, 128, 128),
    }
    state_color = state_colors.get(state, (200, 200, 200))

    cv2.putText(image, f"State: {state}", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, state_color, 2)

    y += 30
    cv2.putText(image, "Head Pose", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    if 'yaw' in details:
        y += 22
        cv2.putText(image, f"  Yaw: {details['yaw']:+.1f} deg", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(image, f"  Pitch: {details['pitch']:+.1f} deg", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(image, f"  Roll: {details['roll']:+.1f} deg", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if 'gaze_yaw' in details:
        y += 28
        cv2.putText(image, "Gaze (Blendshapes)", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        y += 22
        cv2.putText(image, f"  Yaw: {details['gaze_yaw']:+.1f} deg", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(image, f"  Pitch: {details['gaze_pitch']:+.1f} deg", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if 'ear' in details:
        y += 28
        cv2.putText(image, "Eye Openness", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        y += 22
        ear = details['ear']
        ear_color = (0, 255, 0) if ear >= 0.20 else (0, 0, 255)
        cv2.putText(image, f"  EAR: {ear:.3f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_color, 1)

    if 'mar' in details:
        y += 28
        cv2.putText(image, "Mouth Openness (MAR)", (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)
        y += 22
        mar = details['mar']
        mar_color = (0, 255, 0) if mar < 0.35 else (0, 215, 255) if mar < 0.65 else (0, 0, 255)
        cv2.putText(image, f"  MAR: {mar:.3f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, mar_color, 1)

    # FPS
    if 'fps' in details:
        y += 28
        cv2.putText(image, f"FPS: {details['fps']:.1f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return image


def main():
    print("=" * 50)
    print("DMS - Driver Monitoring System")
    print("  Face Mesh  -> MediaPipe Face Landmarker (478 pts)")
    print("  Head Pose  -> solvePnP")
    print("  Gaze       -> MediaPipe Blendshapes")
    print("  Mouth      -> MAR (Mouth Aspect Ratio)")
    print("  Distraction -> Head + Gaze + Drowsiness + Yawning")
    print("=" * 50)
    print("Controls: q=quit, r=reset state, t=toggle landmarks")

    detector = FaceMeshDetector(max_num_faces=1, min_detection_confidence=0.5)
    head_pose = HeadPoseEstimator()
    gaze_estimator = GazeEstimator()
    mouth_detector = MouthDetector()
    dms = DistractionDetector(
        yaw_threshold=25.0,
        pitch_threshold=20.0,
        gaze_yaw_threshold=20.0,
        gaze_pitch_threshold=15.0,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera. Try checking camera index.")
        sys.exit(1)

    print("Camera opened. Running DMS pipeline...")
    print()

    show_all_landmarks = True
    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            face_data = detector.detect(frame)
            landmarks_px = face_data['landmarks_pixel']

            pose = head_pose.estimate(landmarks_px, w, h,
                                      transform=face_data.get('transform'))
            gaze_result = gaze_estimator.estimate(face_data)
            mar = mouth_detector.compute_mar(landmarks_px)
            state, details = dms.update(pose, gaze_result, mar)

            # FPS calculation
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - fps_start + 1e-6)
                fps_start = now
            details['fps'] = fps

            # Visualizations
            if show_all_landmarks:
                detector.draw_landmarks(frame, landmarks_px)
            else:
                # Draw only key landmarks
                key_indices = set(
                    HeadPoseEstimator.DEFAULT_LANDMARK_INDICES +
                    [GazeEstimator.LEFT_IRIS_CENTER, GazeEstimator.RIGHT_IRIS_CENTER]
                )
                detector.draw_landmarks(frame, landmarks_px, key_indices)

            head_pose.draw_axes(frame, landmarks_px, w, h)
            gaze_estimator.draw_gaze(frame, landmarks_px, gaze_result)
            mouth_detector.draw(frame, landmarks_px)
            frame = draw_status_panel(frame, details)

            cv2.imshow("DMS - Driver Monitoring System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                dms.reset()
                print("State reset.")
            elif key == ord('t'):
                show_all_landmarks = not show_all_landmarks
                print(f"Show all landmarks: {show_all_landmarks}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        detector.release()
        cv2.destroyAllWindows()
        print("DMS stopped.")


if __name__ == '__main__':
    main()
