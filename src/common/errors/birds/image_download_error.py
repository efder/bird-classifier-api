from __future__ import annotations

from src.common.errors.base.base_url_error import BaseUrlError


class ImageDownloadError(BaseUrlError):

    @classmethod
    def from_url(cls, url: str) -> ImageDownloadError:
        code: str = 'image_download_error'
        error_message: str = 'There has been an error while downloading the image.'
        return cls(
            code=code,
            error_message=error_message,
            url=url
        )
