import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# ✅ 1. CORS Setup - อนุญาตให้ Frontend ทุกที่เรียกใช้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ✅ 2. LOAD MODELS (Optimized)
# -------------------------
print("🚀 Loading Banana Expert Models...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# พยายามโหลด Model (ถ้า v8s หนักไป ระบบจะสลับไป v8n ให้อัตโนมัติ)
try:
    MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found")
    MODEL_REAL = YOLO(MODEL_PATH)
    print(f"✅ MODEL_REAL: YOLOv8s loaded")
except Exception as e:
    print(f"⚠️ Switching to Fallback (Nano): {e}")
    # ตัว Nano จะเร็วกว่าบน Render (Free Tier) มากครับ
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))

# -------------------------
# ✅ 3. CONFIGURATION
# -------------------------
CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchanggud",
    8: "ngachang", 9: "huamao",
}

async def preprocess_image(file: UploadFile):
    """อ่านภาพ บีบอัดขนาด และเตรียมพร้อมประมวลผล"""
    try:
        # อ่านไฟล์จากบัฟเฟอร์
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # ⚡️ Resize ให้เล็กลง (640px) เพื่อลดการกิน CPU บน Render
            img = cv2.resize(img, (640, 640))
            return img
        return None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

# -------------------------
# ✅ 4. API ROUTES
# -------------------------

@app.get("/")
async def root():
    return {"status": "online", "message": "Banana Expert AI is ready!"}

@app.post("/detect")
@app.post("/detect/") # ✅ รองรับทั้งแบบมีและไม่มี / ปิดท้าย
async def detect(file: UploadFile = File(...)): # ✅ เปลี่ยนจาก image เป็น file ให้ตรงกับ Frontend
    try:
        # 1. อ่านและเตรียมรูป
        img = await preprocess_image(file)
        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # 2. เริ่มการทำนาย (Inference)
        # ปรับความเร็วด้วยการลดขนาด imgsz และปิด verbose
        results = MODEL_REAL.predict(
            source=img, 
            conf=0.15, 
            iou=0.45, 
            imgsz=640, 
            augment=False, 
            verbose=False
        )[0]

        # 3. ตรวจสอบผลลัพธ์
        if not results.boxes or len(results.boxes) == 0:
            return {
                "success": False, 
                "reason": "no_banana_detected"
            }

        # 4. ดึงตัวที่มั่นใจที่สุด
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(confs.argmax())
        
        final_conf = float(confs[best_idx])
        class_id = int(clses[best_idx])
        banana_key = CLASS_KEYS.get(class_id, "unknown")

        # 5. ส่งผลกลับ (ส่งทั้ง banana_key และ class_name เพื่อความชัวร์)
        return {
            "success": True,
            "banana_key": banana_key,
            "class_name": banana_key, # ✅ เพิ่มตัวนี้ให้ตรงกับที่ React เรียก
            "confidence": round(float(final_conf), 3),
            "debug": {
                "count": len(results.boxes),
                "model": "YOLOv8-optimized"
            }
        }

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return {"success": False, "reason": "server_error", "detail": str(e)}

# -------------------------
# ✅ 5. RUN SERVER
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # ใช้สแต็กมาตรฐาน uvicorn เพื่อความเสถียรบน Render
    uvicorn.run(app, host="0.0.0.0", port=port)
