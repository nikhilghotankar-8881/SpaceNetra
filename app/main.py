"""
FastAPI Server Entry Point for SpaceNetra Web Application & GIS Engine.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import logging

logger = logging.getLogger(__name__)

from app.api.endpoints import router as api_router
from app.api.auth_endpoints import router as auth_router
from app.api.websocket_endpoints import router as ws_router
from app.api.metrics_endpoints import router as metrics_router
from app.api.health_endpoints import router as health_router

from app.middleware.security import SecurityHeadersMiddleware, RateLimitMiddleware
from app.middleware.audit import AuditMiddleware
from src.pipeline.integrity import ModelIntegrityVerifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifespan handler.
    Runs SHA-256 model integrity verification on startup.
    """
    logger.info("Initializing SpaceNetra Engine... Running security checks.")
    checkpoint_dir = Path("checkpoints")
    manifest_path = checkpoint_dir / "CHECKSUMS.sha256"
    if manifest_path.exists():
        ok = ModelIntegrityVerifier.verify_manifest(manifest_path, checkpoint_dir)
        if not ok:
            logger.warning("Model integrity verification failed for one or more checkpoints!")
    yield
    logger.info("Shutting down SpaceNetra Engine.")


app = FastAPI(
    title="SpaceNetra Satellite Intelligence Engine",
    description="Operational Satellite Change Detection, Multi-Temporal Series Analysis, Semantic Search, and Confidence Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Registered Security Middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AuditMiddleware)
app.add_middleware(RateLimitMiddleware)

# Restricted CORS Configuration
allowed_origins = os.getenv(
    "SPACENETRA_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Include API Routers
app.include_router(api_router)
app.include_router(auth_router, prefix="/api")
app.include_router(ws_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(health_router, prefix="/api")

# Static directory path setup
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "SpaceNetra Satellite Intelligence Engine",
        "version": "1.0.0",
    }


@app.get("/")
def index():
    """
    Serves SpaceNetra Web Application UI.
    """
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to SpaceNetra Satellite Intelligence Engine API"}
