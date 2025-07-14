from fastapi import APIRouter
from fastapi.responses import FileResponse
import os
from app.schemas.request_response import CoordinateList, ScoringResponse, ScoredCoordinate
from app.ml_models import scoring_service

router = APIRouter()


@router.get("/map")
def get_map():
    file_path = os.path.join("app", "static", "boston_h3_clusters_map.html")
    return FileResponse(file_path, media_type="text/html")


@router.post("/score")
def score_coordinates(data: CoordinateList):
    for idx, group in enumerate(data.root, start=1):
        coords = [(coord.lat, coord.lng) for coord in group.coordinates]
        print(f"route{idx}: {coords}")
    group_results = []
    for group in data.root:
        group_set = []
        for coord in group.coordinates:
            scored = scoring_service.predict_scores([coord.model_dump()])[0]
            group_set.append(ScoredCoordinate(
                lat=coord.lat,
                lng=coord.lng,
                score=scored['score'],
                confidence=scored.get('confidence', 1.0)
            ))
        group_results.append(group_set)
    return group_results


@router.get("/healthcheck")
def healthcheck():
    checks = {
        "status": "ok",
        "model_loaded": scoring_service.model is not None,
        "cluster_map_size": len(scoring_service.cluster_map),
    }
    return checks
