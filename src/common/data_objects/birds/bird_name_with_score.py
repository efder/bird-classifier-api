# pylint: disable=missing-module-docstring
import math

from pydantic import BaseModel


class BirdNameWithScore(BaseModel):
    """
    Basic data object for keeping bird_name and its model score.
    """
    bird_name: str
    score: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BirdNameWithScore):
            return NotImplemented

        return self.bird_name == other.bird_name and (
            math.isclose(self.score, other.score, abs_tol=1e-5)
        )
