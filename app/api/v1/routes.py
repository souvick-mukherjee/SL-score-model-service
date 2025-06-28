from fastapi import APIRouter
from fastapi.responses import FileResponse
import os
from app.schemas.request_response import CoordinateList, ScoringResponse, ScoredCoordinate
from app.ml_models import scoring_service

router = APIRouter()


@router.get("/map")
def get_map():
    file_path = os.path.join("app", "static", "boston_h3_clustered_map.html")
    return FileResponse(file_path, media_type="text/html")


@router.post("/score", response_model=ScoringResponse)
def score_coordinates(data: CoordinateList):
    coords = [coord.dict() for coord in data.coordinates]
    results = scoring_service.predict_scores(coords)
    return {"results": results}


@router.get("/healthcheck")
def healthcheck():
    checks = {
        "status": "ok",
        "model_loaded": scoring_service.model is not None,
        "cluster_map_size": len(scoring_service.cluster_map),
    }
    return checks
