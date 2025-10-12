import mediapipe as mp

import cv2
#MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
holistic=mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Pose Estimation
def pose(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    poseResults = holistic.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(frame, poseResults.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,
                                                     circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                     circle_radius=1))

    mp_drawing.draw_landmarks(frame, poseResults.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                     circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                     circle_radius=2))

    mp_drawing.draw_landmarks(frame, poseResults.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                     circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2,
                                                     circle_radius=2))

    mp_drawing.draw_landmarks(frame, poseResults.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                     circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                     circle_radius=2))
