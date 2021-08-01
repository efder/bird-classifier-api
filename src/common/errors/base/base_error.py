from typing import Any, List, Dict, Optional

from pydantic import BaseModel


class BaseError(BaseModel):
    code: str
    error_message: str

