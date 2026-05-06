"""DMS demo script — runs the full pipeline on sample images (no camera needed).

Usage:
    python demo.py <image_path>            # process a single image
    python demo.py                         # process built-in sample if available
    python demo.py --gen                   # generate a synthetic test face and run
"""

import sys
import os
import cv2
import numpy as np

from face_mesh_detector import FaceMeshDetector
from head_pose_estimator import HeadPoseEstimator
from gaze_estimator import GazeEstimator
from distraction_detector import DistractionDetector, DistractionState


def draw_results(image, pose, gaze_result, state, details, landmarks_px,
                 head_pose_est, gaze_est):
    """Annotate the image with all detection results."""
    h, w = image.shape[:2]
    result = image.copy()

    # Draw key landmarks
    key_indices = set(
        HeadPoseEstimator.DEFAULT_LANDMARK_INDICES +
        [GazeEstimator.LEFT_IRIS_CENTER, GazeEstimator.RIGHT_IRIS_CENTER]
    )
    for i in key_indices:
        if i < len(landmarks_px):
            x, y = landmarks_px[i]
            cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Draw head pose axes
    head_pose_est.draw_axes(result, landmarks_px, w, h)

    # Draw gaze direction
    gaze_est.draw_gaze(result, landmarks_px, gaze_result)

    # Status bar at top
    state_colors = {
        DistractionState.NORMAL.value: (0, 255, 0),
        DistractionState.SLIGHT.value: (0, 215, 255),
        DistractionState.DISTRACTED.value: (0, 0, 255),
        DistractionState.DROWSY.value: (255, 0, 255),
        DistractionState.NO_FACE.value: (128, 128, 128),
    }
    color = state_colors.get(state.value, (200, 200, 200))
    bar_h = 50
    cv2.rectangle(result, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.putText(result, f"State: {state.value}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(result, "DMS Demo", (w - 160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Info panel on the left
    y0 = bar_h + 20
    lines = []
    if 'yaw' in details:
        lines.append(f"Head  | Yaw:{details['yaw']:+.1f}  Pitch:{details['pitch']:+.1f}  Roll:{details['roll']:+.1f}")
    if 'gaze_yaw' in details:
        lines.append(f"Gaze  | Yaw:{details['gaze_yaw']:+.1f}  Pitch:{details['gaze_pitch']:+.1f}")
    if 'ear' in details:
        ear = details['ear']
        lines.append(f"EAR   | {ear:.3f}  {'OPEN' if ear >= 0.20 else 'CLOSED'}")

    for line in lines:
        cv2.putText(result, line, (15, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1)
        y0 += 22

    return result


def generate_synthetic_face(size=(640, 480)):
    """Generate a simple synthetic face image for testing when no image is available."""
    h, w = size
    img = np.full((h, w, 3), (200, 210, 215), dtype=np.uint8)

    face_cx, face_cy = w // 2, h // 2
    # Head oval
    cv2.ellipse(img, (face_cx, face_cy - 20), (120, 150), 0, 0, 360, (180, 200, 210), -1)
    cv2.ellipse(img, (face_cx, face_cy - 20), (120, 150), 0, 0, 360, (140, 160, 170), 2)

    # Eyes
    eye_y = face_cy - 50
    for ex in [face_cx - 40, face_cx + 40]:
        cv2.ellipse(img, (ex, eye_y), (22, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (ex, eye_y), 8, (50, 50, 50), -1)
        cv2.circle(img, (ex - 2, eye_y - 2), 3, (255, 255, 255), -1)

    # Eyebrows
    for bx, by in [(face_cx - 40, eye_y - 20), (face_cx + 40, eye_y - 20)]:
        cv2.ellipse(img, (bx, by), (25, 8), -5, 0, 360, (60, 50, 40), -1)

    # Nose
    nose_tip = (face_cx, face_cy + 20)
    cv2.line(img, (face_cx, face_cy - 10), nose_tip, (130, 110, 100), 3)
    cv2.ellipse(img, nose_tip, (10, 8), 0, 0, 360, (150, 130, 120), -1)

    # Mouth
    mouth_y = face_cy + 60
    cv2.ellipse(img, (face_cx, mouth_y), (30, 12), 0, 0, 180, (160, 100, 100), 2)
    cv2.ellipse(img, (face_cx, mouth_y), (30, 12), 0, 0, 180, (200, 140, 140), -1)

    return img


def process_image(image, detector, head_pose_est, gaze_est, dms):
    """Run the full DMS pipeline on a single image."""
    h, w = image.shape[:2]
    face_data = detector.detect(image)
    landmarks_px = face_data['landmarks_pixel']
    pose = head_pose_est.estimate(landmarks_px, w, h,
                                  transform=face_data.get('transform'))
    gaze_result = gaze_est.estimate(face_data)
    state, details = dms.update(pose, gaze_result)

    # Print results to console
    print("-" * 50)
    print(f"  Image size: {w}x{h}")

    if pose:
        print(f"  Head Pose:   yaw={pose[0]:+.1f}  pitch={pose[1]:+.1f}  roll={pose[2]:+.1f}")
    else:
        print("  Head Pose:   not detected")

    if gaze_result:
        print(f"  Gaze:        yaw={gaze_result['gaze_yaw']:+.1f}  pitch={gaze_result['gaze_pitch']:+.1f}")
        print(f"  EAR:         {gaze_result['avg_ear']:.3f}  (blink L:{gaze_result.get('blink_left',0):.2f} R:{gaze_result.get('blink_right',0):.2f})")
    else:
        print("  Gaze:        not detected")

    print(f"  State:       {state.value}")
    print("-" * 50)

    annotated = draw_results(image, pose, gaze_result, state, details,
                             landmarks_px, head_pose_est, gaze_est)
    return annotated, details


def main():
    print("=" * 55)
    print("  DMS Demo — Sample Image Pipeline")
    print("=" * 55)

    detector = FaceMeshDetector(max_num_faces=1, min_detection_confidence=0.5)
    head_pose_est = HeadPoseEstimator()
    gaze_est = GazeEstimator()
    dms = DistractionDetector()

    image_path = None
    gen_mode = False

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    if '--gen' in flags:
        gen_mode = True
    if args:
        image_path = args[0]

    # Load or generate image
    if image_path and os.path.exists(image_path):
        print(f"\nLoading: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot read image '{image_path}'")
            sys.exit(1)
    elif gen_mode or not image_path:
        print("\nNo image provided. Generating a synthetic test face...")
        image = generate_synthetic_face()
        # Note: synthetic images won't have real MediaPipe detections,
        # so we'll demonstrate the code flow
        print("Note: Synthetic faces may not trigger MediaPipe detection.")
        print("      Provide a real face photo: python demo.py <path>")
    else:
        print(f"\nError: Image not found: '{image_path}'")
        print("Usage: python demo.py <image_path>")
        print("       python demo.py --gen  (use synthetic test image)")
        sys.exit(1)

    annotated, details = process_image(image, detector, head_pose_est, gaze_est, dms)

    # Save result
    output_path = "demo_output.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to: {os.path.abspath(output_path)}")

    # Display result (skip if --no-display flag or no GUI available)
    no_display = '--no-display' in flags
    if no_display:
        print("Skipping display (--no-display flag set).")
    else:
        try:
            cv2.imshow("DMS Demo - Press any key to close", annotated)
            print("Press any key in the image window to exit...")
            cv2.waitKey(0)
        except cv2.error:
            print("No display available (headless mode). Output saved to file.")

    detector.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()
