# pylint: disable=missing-module-docstring
from typing import Any

from pydantic import BaseModel


class ModelRawOutputWithUrl(BaseModel):
    """
    Basic data object for keeping url and related model_raw_output which is used in classification process.
    """
    url: str
    model_raw_output: Any
