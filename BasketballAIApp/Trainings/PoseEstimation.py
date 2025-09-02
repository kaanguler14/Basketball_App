import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import landmark
from mediapipe.tasks.python.core import base_options
from torch.jit import annotate
import time
import threading

class FrameBuffer:
    def __init__(self):
        self.frame =None
        self.lock = threading.Lock()
frame_buffer = FrameBuffer()
stop_flag=False


model_path="D://BasketballAIApp//Models//PoseEstimation//pose_landmarker_full.task"

video_path="D://BasketballAIApp//clips//trainingclips.mp4"

num_poses=4
min_pose_detection_confidence=0.7
min_pose_tracking_confidence=0.7
min_pose_presence_confidence=0.7

def  draw_landmarks(frame, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image=np.copy(frame)

    #loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks=pose_landmarks_list[idx]
        pose_landmarks_proto=landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z
        )for landmark in pose_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto,mp.solutions.pose.POSE_CONNECTIONS,mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
#thread 1
delay=0
def capture_thread():
    global stop_flag
    cap = cv2.VideoCapture(video_path)
    # Videonun orijinal FPS değerini al
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 / video_fps if video_fps > 0 else 0.03
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            print("image capture failed")

            stop_flag=True
            break
        frame=cv2.resize(frame,(640,320))

        with frame_buffer.lock:
            frame_buffer.frame=frame

        time.sleep(delay)  # video gerçek hızında aksın

    cap.release()

#thread 2
def inference_thread():
    global stop_flag
    prev_time=0

    base_options= python.BaseOptions(model_asset_path=model_path)
    options=vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_pose_tracking_confidence,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:

        while not stop_flag:
            with frame_buffer.lock:
                if frame_buffer.frame is None:
                    continue
                frame=frame_buffer.frame.copy()

            mp_image=mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            timestamp_ms= int(cv2.getTickCount() / cv2.getTickFrequency()*1000)
            detection_results=landmarker.detect_for_video(mp_image,timestamp_ms=timestamp_ms)
            annotated_frame=draw_landmarks(frame, detection_results)

            #fps
            curr_time=time.time()
            fps=1/(curr_time-prev_time) if prev_time else 0
            prev_time=curr_time
            cv2.putText(annotated_frame,f"FPS:{fps:2f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.imshow("Pose Estimation", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag=True
                break
    cv2.destroyAllWindows()

#start threads
t1=threading.Thread(target=capture_thread)
t2=threading.Thread(target=inference_thread)

t1.start()
t2.start()

t1.join()
t2.join()
