from typing import Any

from pydantic import BaseModel


class ImageArrayWithUrl(BaseModel):
    url: str
    image_array: Any
