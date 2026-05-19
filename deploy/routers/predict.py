from __future__ import annotations

from datetime import datetime

import requests
from fastapi import APIRouter, File, Form, Request, UploadFile

from deploy.config import HF_API_URL, supabase

router = APIRouter()


def infer_plant_from_disease_label(label: str | None) -> str | None:
    if not isinstance(label, str):
        return None
    if "___" in label:
        plant_part = label.split("___", 1)[0].strip()
        return plant_part or None
    if "_" in label:
        plant_part = label.split("_", 1)[0].strip()
        return plant_part or None
    if " " in label:
        plant_part = label.split(" ", 1)[0].strip()
        return plant_part or None
    return None


@router.post("/predict")
async def predict(
    request: Request, selected_plant: str = Form(...), file: UploadFile = File(...)
):
    try:
        contents = await file.read()

        print(f"🚀 Đang gửi yêu cầu sang Hugging Face cho cây: {selected_plant}")
        response = requests.post(
            HF_API_URL,
            files={"file": (file.filename, contents, file.content_type)},
            data={"selected_plant": selected_plant},
            timeout=120,
        )

        if response.status_code != 200:
            return {"status": "error", "message": "Hugging Face không phản hồi hoặc đang bận"}

        result = response.json()
        if result.get("status") == "error":
            return result

        file_name = f"{datetime.now().timestamp()}.jpg"
        supabase.storage.from_("plant-images").upload(
            file_name, contents, {"content-type": "image/jpeg"}
        )
        image_url = supabase.storage.from_("plant-images").get_public_url(file_name)

        disease_name = result.get("disease")
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
            s = raw_confidence.strip().replace("%", "")
            try:
                confidence_value = float(s)
            except Exception:
                confidence_value = None

        confidence_percent_str = None
        if confidence_value is not None:
            if 0.0 <= confidence_value <= 1.0:
                confidence_percent_value = confidence_value * 100.0
                confidence_percent_str = f"{confidence_percent_value:.2f}%"
            else:
                confidence_percent_str = f"{confidence_value:.2f}%"

        return {
            "status": "success",
            "plant": predicted_plant,
            "disease": disease_name,
            "confidence": confidence_percent_str,
            "image_url": image_url,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
