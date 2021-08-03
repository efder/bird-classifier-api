# pylint: disable=missing-module-docstring
from typing import List

from pydantic import BaseModel, StrictInt, StrictStr


class BirdsClassificationRequestDto(BaseModel):
    """
    Bird Classification Request Data Transfer Object keeps requested bird image urls and
    top_n (how many top results are requested per bird)
    """
    urls: List[StrictStr]
    top_n: StrictInt
