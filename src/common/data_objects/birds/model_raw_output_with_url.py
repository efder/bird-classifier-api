from typing import Any

from pydantic import BaseModel


class ModelRawOutputWithUrl(BaseModel):
    url: str
    model_raw_output: Any
