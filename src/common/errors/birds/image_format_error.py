from __future__ import annotations

from src.common.errors.base.base_url_error import BaseUrlError


class ImageFormatError(BaseUrlError):

    @classmethod
    def from_url(cls, url: str) -> ImageFormatError:
        code: str = 'image_format_error'
        error_message: str = 'There has been an error while formatting the image.'
        return cls(
            code=code,
            error_message=error_message,
            url=url
        )