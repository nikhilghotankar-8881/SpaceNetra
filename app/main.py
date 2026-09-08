"""
FastAPI Server Entry Point for SpaceNetra Web Application & GIS Engine.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router as api_router

app = FastAPI(
    title="SpaceNetra Satellite Intelligence Engine",
    description="Operational Satellite Change Detection, Multi-Temporal Series Analysis, Semantic Search, and Confidence Engine",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

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
