from fastapi import FastAPI

from deploy.routers import health, history, llm, outbreaks, predict, weather

app = FastAPI(title="Plant Disease Detector API")

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(outbreaks.router)
app.include_router(llm.router)
app.include_router(weather.router)
