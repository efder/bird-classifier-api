# pylint: disable=missing-module-docstring
from __future__ import annotations

from src.common.errors.base.base_url_error import BaseUrlError


class ImageDownloadError(BaseUrlError):
    """
    Basic data object to represent any error while downloading an image from the internet.
    """

    @classmethod
    def from_url(cls, url: str) -> ImageDownloadError:
        """
        Creates ImageDownloadError using url
        :param url: requested_url
        :return: error
        """
        code: str = 'image_download_error'
        error_message: str = 'There has been an error while downloading the image.'
        return cls(
            code=code,
            error_message=error_message,
            url=url
        )
