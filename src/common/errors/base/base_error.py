from typing import Any, List, Dict, Optional

from pydantic import BaseModel


class BaseError(BaseModel):
    code: str
    error_message: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseError):
            return NotImplemented
        return self.code == other.code
