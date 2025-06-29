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
    group_results = []
    for group in data.root:
        group_set = []
        for coord in group.coordinates:
            scored = scoring_service.predict_scores([coord.model_dump()])[0]
            group_set.append(f"{{{coord.lat}, {coord.lon}, {scored['score']}}}")
        group_results.append("{" + ",".join(group_set) + "}")
    java_set = "{" + ",".join(group_results) + "}"
    return java_set


@router.get("/healthcheck")
def healthcheck():
    checks = {
        "status": "ok",
        "model_loaded": scoring_service.model is not None,
        "cluster_map_size": len(scoring_service.cluster_map),
    }
    return checks
