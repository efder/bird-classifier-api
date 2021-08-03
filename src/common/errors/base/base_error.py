# pylint: disable=missing-module-docstring
from pydantic import BaseModel


class BaseError(BaseModel):
    """
    Basic data object for keeping errors with its code and error message.
    """
    code: str
    error_message: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseError):
            return NotImplemented
        return self.code == other.code
