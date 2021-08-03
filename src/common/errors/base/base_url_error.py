from src.common.errors.base.base_error import BaseError


class BaseUrlError(BaseError):
    url: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseUrlError):
            return NotImplemented

        return super().__eq__(other) and self.url == other.url
