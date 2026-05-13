import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import create_all_tables
from app.routes import auth, screening, reports, reminders, admin, eye_scan
import app.models  # noqa: F401

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MNR Eye Health Platform starting...")
    create_all_tables()
    logger.info("Database tables verified/created")
    os.makedirs("generated_reports", exist_ok=True)
    logger.info(f"App: {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"AI Model: {settings.GEMINI_MODEL}")
    yield
    logger.info("MNR Eye Health Platform shutting down")


app = FastAPI(
    title="MNR Eye Health Platform API",
    description="AI-Powered Digital Eye Health Platform",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list + [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."}
    )


app.include_router(auth.router)
app.include_router(screening.router)
app.include_router(reports.router)
app.include_router(reminders.router)
app.include_router(admin.router)
app.include_router(eye_scan.router)


@app.get("/", tags=["System"])
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
def health_check():
    from app.core.database import engine
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "ai_configured": bool(
            settings.GEMINI_API_KEY and
            settings.GEMINI_API_KEY not in ("", "dummy-key-replace-with-real-key")
        ),
    }