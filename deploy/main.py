from fastapi import FastAPI
from fastapi.responses import JSONResponse

from deploy.routers import auth_pages, consultations, health, history, llm, outbreaks, predict, weather
from deploy.security import security_middleware

app = FastAPI(title="Plant Disease Detector API")
app.middleware("http")(security_middleware)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_, exc: RuntimeError):
    message = str(exc)
    if "Missing required environment variable" in message or "Missing DB_" in message:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": message,
                "hint": "Set SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DB_USER, DB_PASSWORD on Render.",
            },
        )
    return JSONResponse(status_code=500, content={"status": "error", "message": message})


app.include_router(auth_pages.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(consultations.router)
app.include_router(outbreaks.router)
app.include_router(llm.router)
app.include_router(weather.router)
