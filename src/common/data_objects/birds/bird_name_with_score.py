from pydantic import BaseModel


class BirdNameWithScore(BaseModel):
    bird_name: str
    score: float
