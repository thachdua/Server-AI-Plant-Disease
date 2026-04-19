import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from supabase import create_client, Client
import psycopg2
from datetime import datetime

app = FastAPI()

# --- 1. SUPABASE CONFIG ---
SUPABASE_URL = "https://wxmmfmvyefruyknymvdm.supabase.co"
SUPABASE_KEY = "sb_publishable_gW6BTa8IWvVjO6Rbe-OGxQ_WgD6knWz"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. DATABASE CONFIG (🔥 ĐÃ FIX) ---
DB_CONFIG = {
    "host": "db.wxmmfmvyefruyknymvdm.supabase.co",  # ✅ direct DB
    "database": "postgres",
    "user": "postgres",
    "password": "Nguyenyeuloc@123",  # ✅ password thật của bạn
    "port": 5432,
    "sslmode": "require"
}

# --- 3. SAVE TO DATABASE ---
def save_to_db(plant_name, disease_name, confidence, image_url):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = """
        INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """

        cur.execute(query, (
            plant_name,
            disease_name,
            confidence,
            image_url,
            datetime.now()
        ))

        conn.commit()
        cur.close()
        conn.close()

        print("✅ ĐÃ LƯU DATABASE")

    except Exception as e:
        print(f"❌ Lỗi Database: {e}")

# --- 4. LOAD MODEL ---
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(
            "model_strong_cnn_part1.keras",
            custom_objects={'swish': tf.nn.swish}
        )
        print("✅ MODEL READY")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")

# --- 5. CLASS NAMES ---
CLASS_NAMES = [
    'apple___alternaria_leaf_spot', 'apple___black_rot', 'apple___brown_spot', 'apple___gray_spot', 'apple___healthy', 'apple___rust', 'apple___scab', 
    'bell_pepper___bacterial_spot', 'bell_pepper___healthy', 
    'blueberry___healthy', 
    'cassava___bacterial_blight', 'cassava___brown_streak_disease', 'cassava___green_mottle', 'cassava___healthy', 'cassava___mosaic_disease', 
    'cherry___healthy', 'cherry___powdery_mildew', 
    'coffee___healthy', 'coffee___red_spider_mite', 'coffee___rust', 
    'corn___common_rust', 'corn___gray_leaf_spot', 'corn___healthy', 'corn___northern_leaf_blight', 
    'grape___black_measles', 'grape___black_rot', 'grape___healthy', 'grape___leaf_blight', 
    'orange___citrus_greening', 
    'peach___bacterial_spot', 'peach___healthy', 
    'potato___bacterial_wilt', 'potato___early_blight', 'potato___healthy', 'potato___late_blight', 'potato___leafroll_virus', 'potato___mosaic_virus', 'potato___nematode', 'potato___pests', 'potato___phytophthora', 
    'raspberry___healthy', 
    'rice___bacterial_blight', 'rice___blast', 'rice___brown_spot', 'rice___tungro', 
    'rose___healthy', 'rose___rust', 'rose___slug_sawfly', 
    'soybean___healthy', 
    'squash___powdery_mildew', 
    'strawberry___healthy', 'strawberry___leaf_scorch', 
    'sugercane___healthy', 'sugercane___mosaic', 'sugercane___red_rot', 'sugercane___rust', 'sugercane___yellow_leaf', 
    'tomato___bacterial_spot', 'tomato___early_blight', 'tomato___healthy', 'tomato___late_blight', 'tomato___leaf_curl', 'tomato___leaf_mold', 'tomato___mosaic_virus', 'tomato___septoria_leaf_spot', 'tomato___spider_mites', 'tomato___target_spot', 
    'watermelon___anthracnose', 'watermelon___downy_mildew', 'watermelon___healthy', 'watermelon___mosaic_virus'
]

# --- 6. API PREDICT ---
@app.post("/predict")
async def predict(selected_plant: str = Form(...), file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]

        valid_indices = [
            i for i, name in enumerate(CLASS_NAMES)
            if name.split('___')[0].title() == selected_plant.title()
        ]

        if not valid_indices:
            raise HTTPException(status_code=400, detail="Loại cây không hợp lệ")

        valid_probs = {i: predictions[i] for i in valid_indices}
        top1_idx = max(valid_probs, key=valid_probs.get)

        disease_full = CLASS_NAMES[top1_idx].replace('___', ' - ').replace('_', ' ').title()
        confidence = float(valid_probs[top1_idx] * 100)

        # --- Upload ảnh ---
        file_name = f"{datetime.now().timestamp()}.jpg"
        supabase.storage.from_("plant-images").upload(
            file_name,
            contents,
            {"content-type": "image/jpeg"}
        )

        image_url = supabase.storage.from_("plant-images").get_public_url(file_name)

        # --- Save DB ---
        save_to_db(selected_plant, disease_full, confidence, image_url)

        return {
            "status": "success",
            "plant": selected_plant,
            "disease": disease_full,
            "confidence": f"{confidence:.2f}%",
            "image_url": image_url
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 7. RUN ---
if __name__ == "__main__":
    import uvicorn
    import os # Nhớ đảm bảo đã có import os ở đầu file
    # Lấy port từ Render cấp, nếu chạy local thì mặc định là 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)