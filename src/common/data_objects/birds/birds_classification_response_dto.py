from typing import Dict, List, Optional, Any

from pydantic import BaseModel

from src.common.data_objects.birds.bird_name_with_score import BirdNameWithScore


class BirdsClassificationResponseDto(BaseModel):
    data: Optional[Dict[str, List[BirdNameWithScore]]]
    errors: Optional[Dict[str, Any]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BirdsClassificationResponseDto):
            return NotImplemented

        return self.data == other.data and self.errors == other.errors
