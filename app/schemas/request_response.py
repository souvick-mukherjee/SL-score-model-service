from pydantic import BaseModel
from typing import List


class Coordinate(BaseModel):
    lat: float
    lon: float


class CoordinateList(BaseModel):
    coordinates: List[Coordinate]


class ScoredCoordinate(BaseModel):
    lat: float
    lon: float
    score: int
    confidence: float


class ScoringResponse(BaseModel):
    results: List[ScoredCoordinate]
