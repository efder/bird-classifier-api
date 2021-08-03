# pylint: disable=missing-module-docstring
from src.common.errors.base.base_error import BaseError


class BaseUrlError(BaseError):
    """
    Basic data object for keeping errors with their related file urls.
    """
    url: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseUrlError):
            return NotImplemented

        return super().__eq__(other) and self.url == other.url
