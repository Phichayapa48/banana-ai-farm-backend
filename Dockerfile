FROM python:3.11-slim

# 1. ติดตั้ง System Dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. ติดตั้ง Library (ใช้ Cache เพื่อความเร็ว)
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

# 3. ก๊อปปี้ไฟล์ทั้งหมด (เช็คว่ามีโฟลเดอร์ model และไฟล์ app.py อยู่ด้วยนะ)
COPY . .

# 4. ตั้งค่า Env กัน Python งอแง
ENV PYTHONUNBUFFERED=1

# 5. รัน Server (ใช้พอร์ต 10000 ตามที่แอ๋มกำหนดในโค้ด)
# บรรทัดนี้ต้องเป็น app:app เพราะไฟล์ชื่อ app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
