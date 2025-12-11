"""
Basketball Shot Analyzer - Web API
==================================
FastAPI backend for video upload and shot analysis.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
import threading
import json
import cv2
import numpy as np
from pathlib import Path
from io import BytesIO

from processor import VideoProcessor

# Paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(
    title="Basketball Shot Analyzer",
    description="Upload a basketball video and get shot analysis",
    version="1.0.0"
)

# CORS - Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Store processing status
processing_status = {}


@app.get("/")
async def root():
    """Serve frontend"""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    num_players: int = 1
):
    """
    Upload a video - returns job_id and first frame for 3PT line selection.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(400, "Only video files allowed (.mp4, .avi, .mov, .mkv)")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize status - waiting for 3PT selection
    processing_status[job_id] = {
        "status": "awaiting_calibration",
        "progress": 0,
        "message": "3 sayı çizgisini seçin",
        "result": None,
        "video_path": str(video_path),
        "num_players": num_players
    }
    
    return {"job_id": job_id, "message": "Video yüklendi, 3PT çizgisini seçin"}


@app.get("/api/preview/{job_id}")
async def get_preview_frame(job_id: str):
    """Get first frame of video for calibration"""
    if job_id not in processing_status:
        raise HTTPException(404, "Job not found")
    
    video_path = processing_status[job_id].get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(404, "Video not found")
    
    # Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(500, "Could not read video")
    
    # Resize for display
    scale = 0.5
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return StreamingResponse(
        BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


@app.post("/api/start-process/{job_id}")
async def start_processing(
    job_id: str,
    three_point_line: str = Form(...)
):
    """Start processing with user-defined 3PT line"""
    if job_id not in processing_status:
        raise HTTPException(404, "Job not found")
    
    status = processing_status[job_id]
    if status["status"] != "awaiting_calibration":
        raise HTTPException(400, "Job already started or completed")
    
    # Parse 3PT line points
    try:
        points = json.loads(three_point_line)
        if len(points) < 3:
            raise HTTPException(400, "En az 3 nokta gerekli")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON format")
    
    # Update status
    processing_status[job_id]["status"] = "processing"
    processing_status[job_id]["progress"] = 5
    processing_status[job_id]["message"] = "İşlem başlıyor..."
    processing_status[job_id]["three_point_line"] = points
    
    video_path = status["video_path"]
    num_players = status["num_players"]
    
    # Start processing in background
    thread = threading.Thread(
        target=process_video_task,
        args=(job_id, video_path, num_players, points),
        daemon=True
    )
    thread.start()
    
    return {"message": "İşlem başladı"}


def process_video_task(job_id: str, video_path: str, num_players: int, three_point_line: list = None):
    """Background task for video processing"""
    print(f"[TASK] Starting processing for job {job_id}")
    print(f"[TASK] 3PT Line: {three_point_line}")
    try:
        processor = VideoProcessor(
            video_path=video_path,
            num_players=num_players,
            output_dir=str(OUTPUT_DIR),
            job_id=job_id,
            status_callback=lambda msg, prog: update_status(job_id, msg, prog),
            three_point_line=three_point_line
        )
        
        print(f"[TASK] Processor created, starting process...")
        result = processor.process()
        print(f"[TASK] Processing complete!")
        
        processing_status[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "İşlem tamamlandı!",
            "result": result
        }
        
    except Exception as e:
        import traceback
        print(f"[TASK ERROR] {str(e)}")
        traceback.print_exc()
        processing_status[job_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Hata: {str(e)}",
            "result": None
        }
    
    finally:
        # Cleanup uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)


def update_status(job_id: str, message: str, progress: int):
    """Update processing status"""
    if job_id in processing_status:
        processing_status[job_id]["message"] = message
        processing_status[job_id]["progress"] = progress


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status"""
    if job_id not in processing_status:
        raise HTTPException(404, "Job not found")
    return processing_status[job_id]


@app.get("/api/video/{job_id}")
async def get_processed_video(job_id: str):
    """Get processed video"""
    video_path = OUTPUT_DIR / f"{job_id}_output.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@app.delete("/api/cleanup/{job_id}")
async def cleanup(job_id: str):
    """Cleanup job files"""
    # Remove from status
    if job_id in processing_status:
        del processing_status[job_id]
    
    # Remove output file
    video_path = OUTPUT_DIR / f"{job_id}_output.mp4"
    if video_path.exists():
        os.remove(video_path)
    
    return {"message": "Cleanup completed"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 6000-6063 portları tarayıcılarda engelli

