from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "FastAPI backend çalışıyor!"}


# ================== MODELLER ==================
# YOLO: Basketbol + Pota
yolo_model = YOLO("D://repos//Basketball_App//BasketballAIApp//Trainings//kagglebest.pt")

# Mediapipe Pose
MODEL_PATH = "D://BasketballAIApp//Models//PoseEstimation//pose_landmarker_full.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=4,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)


# ================== ENDPOINT ==================
@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    # Dosyayı oku
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w, _ = img.shape

    # -------- YOLO İLE BASKETBOL + POTA --------
    objects = {"basketball": None, "hoop": None}
    results = yolo_model(img)
    for r in results[0].boxes:
        cls_id = int(r.cls[0])
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cls_id == 0:  # basketball
            objects["basketball"] = {"x": cx, "y": cy}
        elif cls_id == 1:  # hoop
            objects["hoop"] = {"x": cx, "y": cy}

    # -------- MEDIAPIPE İLE OYUNCULAR --------
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )
    detection_result = pose_landmarker.detect(mp_image)

    players = []
    for idx, landmarks in enumerate(detection_result.pose_landmarks):
        keypoints = {}
        for lm_id, lm in enumerate(landmarks):
            keypoints[lm_id] = {
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "z": lm.z
            }
        players.append({"id": idx, "keypoints": keypoints})

    # -------- SONUÇ JSON --------
    return JSONResponse(content={
        "objects": objects,
        "players": players
    })
