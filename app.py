import os
import cv2
import numpy as np
import gc  # เพิ่มเข้ามาเพื่อเคลียร์ RAM
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# 1. CORS - ปรับให้ครอบคลุมและปลอดภัย
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD MODEL - เช็ค Path ให้ชัวร์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

print("🚀 Starting Banana Expert AI...")
try:
    # พยายามโหลดรุ่น Small ก่อน ถ้าไม่ได้ค่อยไป Nano
    MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8nbg.pt")
    
    MODEL_REAL = YOLO(MODEL_PATH)
    print(f"✅ Model loaded: {os.path.basename(MODEL_PATH)}")
except Exception as e:
    print(f"❌ Critical Error: Could not load model: {e}")
    # ป้องกัน App พังตอนรัน ให้ใส่ตัวแปรว่างไว้ก่อน
    MODEL_REAL = None

CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchangkut",
    8: "ngachang", 9: "huamao",
}

@app.get("/")
async def root():
    return {
        "status": "online", 
        "model_loaded": MODEL_REAL is not None,
        "message": "AI Server is ready to peel!"
    }

@app.post("/detect")
@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    if MODEL_REAL is None:
        return {"success": False, "reason": "model_not_ready"}

    try:
        # อ่านไฟล์รูป
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # AI Prediction
        # ปรับ imgsz เป็น 640 ตามที่เทรนมา
        results = MODEL_REAL.predict(
            source=img,
            conf=0.20,  # เพิ่มนิดหน่อยเพื่อลด False Positive
            iou=0.45,
            imgsz=640,
            save=False,
            verbose=False
        )[0]

        if not results.boxes or len(results.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        # ดึงตัวที่มั่นใจที่สุด (Best Confidence)
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(np.argmax(confs))
        
        raw_slug = CLASS_KEYS.get(int(clses[best_idx]), "unknown")
        
        # คืนค่ากลับไป
        return {
            "success": True,
            "banana_key": raw_slug,
            "confidence": round(float(confs[best_idx]), 4),
            "debug": {
                "detected_count": len(results.boxes),
                "model": "YOLOv8-optimized"
            }
        }

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return {"success": False, "reason": "internal_server_error", "detail": str(e)}
    
    finally:
        # สำคัญมาก: เคลียร์ Memory ป้องกัน RAM เต็มบน Server
        if 'img' in locals(): del img
        if 'results' in locals(): del results
        gc.collect() 
        await file.close()

if __name__ == "__main__":
    # Render มักจะต้องการให้รันผ่าน PORT env
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
