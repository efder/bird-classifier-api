# pylint: disable=missing-module-docstring
from typing import Any

from pydantic import BaseModel


class ImageArrayWithUrl(BaseModel):
    """
    Basic data object which keeps bird image url and its image as numpy array
    """
    url: str
    image_array: Any
