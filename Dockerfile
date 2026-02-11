FROM python:3.11-slim

# 1. ติดตั้ง System Dependencies (แก้ชื่อแพ็กเกจแล้ว)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. ติดตั้ง Library
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

# 3. ก๊อปปี้โค้ดและโมเดล
COPY . .

ENV PYTHONUNBUFFERED=1

# 4. รัน Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
