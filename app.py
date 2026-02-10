import os
import cv2
import numpy as np
import gc
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# ✅ CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ในโปรดักชั่นควรระบุ Domain Vercel ของพี่นะครับ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Model Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_REAL = None

CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchangkut",
    8: "ngachang", 9: "huamao"
}

# ✅ Load Model ทันทีที่เครื่องเปิด (Startup)
@app.on_event("startup")
def load_model():
    global MODEL_REAL
    try:
        model_files = ["best_modelv8sbg.pt", "best_modelv8nbg.pt"]
        found_path = None
        for f in model_files:
            p = os.path.join(MODEL_DIR, f)
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            MODEL_REAL = YOLO(found_path)
            print(f"✅ Model loaded successfully: {found_path}")
        else:
            print("❌ No model file found!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/")
def root():
    return {"status": "online", "model_ready": MODEL_REAL is not None}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if MODEL_REAL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. อ่านไฟล์รูป
        contents = await file.read()
        
        # 2. แปลงเป็น OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # 3. Predict (ใช้ torch.no_grad() เพื่อประหยัด RAM)
        with torch.no_grad():
            results = MODEL_REAL.predict(
                source=img,
                conf=0.25,
                imgsz=640,
                verbose=False
            )[0]

        # 4. ตรวจสอบผล
        if not results.boxes or len(results.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(np.argmax(confs))

        return {
            "success": True,
            "banana_key": CLASS_KEYS.get(clses[best_idx], "unknown"),
            "confidence": float(confs[best_idx])
        }

    except Exception as e:
        return {"success": False, "reason": str(e)}
    
    finally:
        await file.close()
        # เคลียร์ RAM หลังจบงาน
        if 'img' in locals(): del img
        if 'contents' in locals(): del contents
        gc.collect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
