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

        # 4. Chuẩn hoá kết quả từ Hugging Face
        #    Lưu ý: trước đây API đang luôn trả plant = selected_plant (input),
        #    khiến app thấy "Tomato" hoài nếu UI gửi mặc định Tomato.
        disease_name = result.get("disease")

        def infer_plant_from_disease_label(label: str | None) -> str | None:
            if not isinstance(label, str):
                return None
            # common formats:
            # - "Apple___healthy"
            # - "Tomato___Late_blight"
            # - "Apple healthy" (fallback)
            if "___" in label:
                plant_part = label.split("___", 1)[0].strip()
                return plant_part or None
            if "_" in label:
                # if model returns "Apple_healthy" (single underscore)
                plant_part = label.split("_", 1)[0].strip()
                return plant_part or None
            if " " in label:
                plant_part = label.split(" ", 1)[0].strip()
                return plant_part or None
            return None

        predicted_plant = (
            result.get("plant")
            or infer_plant_from_disease_label(disease_name)
            or selected_plant
        )

        raw_confidence = result.get("confidence")
        confidence_value = None
        if isinstance(raw_confidence, (int, float)):
            confidence_value = float(raw_confidence)
        elif isinstance(raw_confidence, str):
            # chấp nhận "0.87", "87", "87%" ...
            s = raw_confidence.strip().replace("%", "")
            try:
                confidence_value = float(s)
            except Exception:
                confidence_value = None

        # chuẩn hoá về chuỗi phần trăm
        # - nếu model trả [0..1] thì đổi sang %
        # - nếu đã là [0..100] thì giữ nguyên
        confidence_percent_str = None
        confidence_percent_value = None
        if confidence_value is not None:
            if 0.0 <= confidence_value <= 1.0:
                confidence_percent_value = confidence_value * 100.0
                confidence_percent_str = f"{confidence_percent_value:.2f}%"
            else:
                confidence_percent_value = confidence_value
                confidence_percent_str = f"{confidence_percent_value:.2f}%"

        # 5. Lưu vào Database
        # DB column is typically numeric/double precision → store number (0..100), not "xx.xx%"
        save_to_db(predicted_plant, disease_name, confidence_percent_value, image_url)

        # 6. Trả kết quả cuối cùng về cho iOS App
        return {
            "status": "success",
            "plant": predicted_plant,
            "disease": disease_name,
            "confidence": confidence_percent_str,
            "image_url": image_url
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)