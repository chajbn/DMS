"""Distraction detection combining head pose and gaze signals."""

import time
from enum import Enum


class DistractionState(Enum):
    NORMAL = "Normal"
    SLIGHT = "Slight Distraction"
    DISTRACTED = "Distracted"
    DROWSY = "Drowsy"
    NO_FACE = "No Face"


class DistractionDetector:
    """Combines head pose and gaze to determine driver distraction state."""

    def __init__(self,
                 yaw_threshold=25.0,
                 pitch_threshold=20.0,
                 gaze_yaw_threshold=20.0,
                 gaze_pitch_threshold=15.0,
                 ear_drowsy_threshold=0.20,
                 mar_yawn_threshold=0.65,
                 drowsy_duration=1.5,
                 distracted_duration=1.0,
                 yawn_duration=1.0):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.gaze_yaw_threshold = gaze_yaw_threshold
        self.gaze_pitch_threshold = gaze_pitch_threshold
        self.ear_drowsy_threshold = ear_drowsy_threshold
        self.mar_yawn_threshold = mar_yawn_threshold
        self.drowsy_duration = drowsy_duration
        self.distracted_duration = distracted_duration
        self.yawn_duration = yawn_duration

        self._distracted_start = None
        self._drowsy_start = None
        self._yawn_start = None
        self._current_state = DistractionState.NORMAL

    def update(self, head_pose, gaze_result, mar=None):
        """Update distraction state based on head pose and gaze.

        Args:
            head_pose: (yaw, pitch, roll) or None
            gaze_result: dict from GazeEstimator.estimate() or None
            mar: Mouth Aspect Ratio (float or None)

        Returns:
            (DistractionState, details_dict)
        """
        now = time.time()

        if head_pose is None or gaze_result is None:
            self._distracted_start = None
            self._drowsy_start = None
            self._yawn_start = None
            self._current_state = DistractionState.NO_FACE
            return self._current_state, self._get_details(head_pose, gaze_result, mar)

        yaw, pitch, roll = head_pose
        gaze_yaw = gaze_result['gaze_yaw']
        gaze_pitch = gaze_result['gaze_pitch']
        avg_ear = gaze_result['avg_ear']

        # Check drowsiness (eye closure)
        if avg_ear < self.ear_drowsy_threshold:
            if self._drowsy_start is None:
                self._drowsy_start = now
            drowsy_elapsed = now - self._drowsy_start
            if drowsy_elapsed >= self.drowsy_duration:
                self._current_state = DistractionState.DROWSY
                return self._current_state, self._get_details(head_pose, gaze_result, mar)
        else:
            self._drowsy_start = None

        # Check yawning (mouth openness as drowsiness signal)
        if mar is not None and mar > self.mar_yawn_threshold:
            if self._yawn_start is None:
                self._yawn_start = now
            yawn_elapsed = now - self._yawn_start
            if yawn_elapsed >= self.yawn_duration:
                self._current_state = DistractionState.DROWSY
                return self._current_state, self._get_details(head_pose, gaze_result, mar)
        else:
            self._yawn_start = None

        # Check distraction (head pose + gaze deviation)
        head_distracted = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold
        gaze_distracted = abs(gaze_yaw) > self.gaze_yaw_threshold or abs(gaze_pitch) > self.gaze_pitch_threshold

        if head_distracted and gaze_distracted:
            if self._distracted_start is None:
                self._distracted_start = now
            elapsed = now - self._distracted_start
            if elapsed >= self.distracted_duration:
                self._current_state = DistractionState.DISTRACTED
            else:
                self._current_state = DistractionState.SLIGHT
        elif head_distracted or gaze_distracted:
            if self._distracted_start is None:
                self._distracted_start = now
            elapsed = now - self._distracted_start
            if elapsed >= self.distracted_duration * 1.5:
                self._current_state = DistractionState.SLIGHT
            else:
                self._current_state = DistractionState.NORMAL
        else:
            self._distracted_start = None
            self._current_state = DistractionState.NORMAL

        return self._current_state, self._get_details(head_pose, gaze_result, mar)

    def _get_details(self, head_pose, gaze_result, mar=None):
        details = {'state': self._current_state.value}

        if head_pose:
            yaw, pitch, roll = head_pose
            details.update({
                'yaw': round(yaw, 1),
                'pitch': round(pitch, 1),
                'roll': round(roll, 1),
            })

        if gaze_result:
            details.update({
                'gaze_yaw': round(gaze_result['gaze_yaw'], 1),
                'gaze_pitch': round(gaze_result['gaze_pitch'], 1),
                'ear': round(gaze_result['avg_ear'], 3),
            })

        if mar is not None:
            details['mar'] = round(mar, 3)

        return details

    def reset(self):
        self._distracted_start = None
        self._drowsy_start = None
        self._yawn_start = None
        self._current_state = DistractionState.NORMAL
