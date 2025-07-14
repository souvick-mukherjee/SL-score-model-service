from pydantic import BaseModel, RootModel
from typing import List


class Coordinate(BaseModel):
    lat: float
    lng: float

class CoordinateGroup(BaseModel):
    coordinates: List[Coordinate]

class CoordinateList(RootModel[list[CoordinateGroup]]):
    pass


class ScoredCoordinate(BaseModel):
    lat: float
    lng: float
    score: int
    confidence: float


class ScoringResponse(BaseModel):
    results: List[ScoredCoordinate]
