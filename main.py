import os
import io
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from supabase import create_client, Client
import psycopg2
from datetime import datetime

app = FastAPI()

# --- 1. CONFIG ---
# Link API Hugging Face mà bạn vừa tạo
HF_API_URL = "https://thachdua-plantdiseasedectect.hf.space/predict"

SUPABASE_URL = "https://wxmmfmvyefruyknymvdm.supabase.co"
SUPABASE_KEY = "sb_publishable_gW6BTa8IWvVjO6Rbe-OGxQ_WgD6knWz"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Config DB dùng Pooler (Cổng 6543) cho ổn định trên Cloud
DB_CONFIG = {
    "host": "aws-1-ap-southeast-1.pooler.supabase.com",
    "database": "postgres",
    "user": "postgres.wxmmfmvyefruyknymvdm",
    "password": os.environ.get("DB_PASSWORD", "Nguyenyeuloc@123"),
    "port": 6543,
    "sslmode": "require"
}

# --- 2. HÀM LƯU DATABASE ---
def save_to_db(plant_name, disease_name, confidence, image_url):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        query = """
        INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (plant_name, disease_name, confidence, image_url, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Đã lưu lịch sử vào Supabase")
    except Exception as e:
        print(f"❌ Lỗi lưu DB: {e}")

@app.get("/")
def home():
    return {"message": "Render Bridge is Online", "target": "Hugging Face Space"}

# --- 3. API CHÍNH ---
@app.post("/predict")
async def predict(selected_plant: str = Form(...), file: UploadFile = File(...)):
    try:
        # 1. Đọc file ảnh
        contents = await file.read()

        # 2. Gửi ảnh sang Hugging Face để AI xử lý (Dùng 16GB RAM bên đó)
        print(f"🚀 Đang gửi yêu cầu sang Hugging Face cho cây: {selected_plant}")
        response = requests.post(
            HF_API_URL,
            files={"file": (file.filename, contents, file.content_type)},
            data={"selected_plant": selected_plant},
            timeout=120 # Đợi tối đa 2 phút nếu HF đang thức dậy
        )
        
        if response.status_code != 200:
            return {"status": "error", "message": "Hugging Face không phản hồi hoặc đang bận"}

        result = response.json()
        if result.get("status") == "error":
            return result

        # 3. Upload ảnh lên Supabase Storage như cũ
        file_name = f"{datetime.now().timestamp()}.jpg"
        supabase.storage.from_("plant-images").upload(
            file_name, contents, {"content-type": "image/jpeg"}
        )
        image_url = supabase.storage.from_("plant-images").get_public_url(file_name)

        # 4. Lưu vào Database
        disease_name = result.get("disease")
        confidence = result.get("confidence")
        save_to_db(selected_plant, disease_name, confidence, image_url)

        # 5. Trả kết quả cuối cùng về cho iOS App
        return {
            "status": "success",
            "plant": selected_plant,
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "image_url": image_url
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)