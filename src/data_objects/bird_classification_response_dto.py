from typing import Dict, List

from pydantic import BaseModel

from src.data_objects.bird_name_with_score import BirdNameWithScore
from src.errors.base_error import BaseError


class BirdClassificationResponseDto(BaseModel):
    data: Dict[str, List[BirdNameWithScore]]
    errors: Dict[str, BaseError]
