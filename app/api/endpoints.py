"""
SpaceNetra REST API Endpoints Router.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
import numpy as np
from pydantic import BaseModel

from src.confidence.quality_engine import ConfidenceEngine
from src.pipeline.sentinel_cd import SentinelChangeDetector
from src.retrieval.semantic_search import SemanticSearchEngine
from src.temporal.multitemporal import MultiTemporalAnalyzer

router = APIRouter(prefix="/api", tags=["SpaceNetra API"])


class PredictRequest(BaseModel):
    architecture: str = "siamese_unet"
    threshold: float = 0.5
    dry_run: bool = True


class SearchRequest(BaseModel):
    query: str = "solar panels in desert"
    top_k: int = 5


class ScoreRequest(BaseModel):
    dry_run: bool = True


@router.post("/predict")
def predict_change(req: PredictRequest) -> Dict[str, Any]:
    """
    Executes Satellite Pair Change Detection.
    """
    try:
        detector = SentinelChangeDetector(model_or_arch=req.architecture)
        if req.dry_run:
            t1 = np.random.randint(500, 4000, (256, 256, 3), dtype=np.uint16)
            t2 = np.random.randint(500, 4000, (256, 256, 3), dtype=np.uint16)
            res = detector.predict_pair(t1, t2, threshold=req.threshold)
            prob_map = res["probability_map"]
            change_mask = res["change_mask"]

            return {
                "status": "success",
                "architecture": req.architecture,
                "threshold": req.threshold,
                "probability_map_shape": list(prob_map.shape),
                "prob_min": float(prob_map.min()),
                "prob_max": float(prob_map.max()),
                "change_pixels": int(np.sum(change_mask > 0)),
            }
        else:
            return {"status": "error", "message": "Provide file upload for real inference"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multitemporal")
def multitemporal_analysis(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Multi-Temporal Time-Series Analysis.
    """
    try:
        analyzer = MultiTemporalAnalyzer()
        scenes = [np.random.randint(500, 4000, (256, 256, 3), dtype=np.uint16) for _ in range(4)]
        timestamps = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]

        res = analyzer.analyze_series(scene_list=scenes, timestamps=timestamps)

        return {
            "status": "success",
            "scenes_processed": len(scenes),
            "step_transitions": len(res["step_transitions"]),
            "baseline_transitions": len(res["baseline_transitions"]),
            "cumulative_max_frequency": int(res["cumulative_frequency"].max()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_tiles(req: SearchRequest) -> Dict[str, Any]:
    """
    Executes Natural Language Semantic Tile Search.
    """
    try:
        engine = SemanticSearchEngine()
        # Seed index with synthetic tiles for demonstration
        patches = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(5)]
        metadata = [{"tile_id": f"tile_{i:02d}", "region": f"Zone_{i}"} for i in range(5)]
        engine.index_patches(patches, metadata)

        results = engine.search_by_text(req.query, top_k=req.top_k)

        return {
            "status": "success",
            "query": req.query,
            "results_count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score")
def score_quality(req: ScoreRequest) -> Dict[str, Any]:
    """
    Evaluates Quality Confidence for Satellite Scene Pair.
    """
    try:
        engine = ConfidenceEngine()
        t1 = np.random.randint(500, 4000, (256, 256, 3), dtype=np.uint16)
        t2 = t1 + np.random.randint(-100, 100, (256, 256, 3), dtype=np.int16)
        t2 = np.clip(t2, 0, 10000).astype(np.uint16)
        prob_map = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)

        res = engine.evaluate_quality(img_t1=t1, img_t2=t2, prob_map=prob_map)

        return {
            "status": "success",
            "quality_report": res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
