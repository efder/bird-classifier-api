from typing import List

from pydantic import BaseModel, StrictInt, StrictStr


class BirdsClassificationRequestDto(BaseModel):
    urls: List[StrictStr]
    top_n: StrictInt
