# pylint: disable=missing-module-docstring
from __future__ import annotations

from src.common.errors.base.base_url_error import BaseUrlError


class ImageFormatError(BaseUrlError):
    """
    Basic data object to represent any error while formatting a file to make it compatible with the model.
    """

    @classmethod
    def from_url(cls, url: str) -> ImageFormatError:
        """
        Creates ImageFormatError using url
        :param url: requested_url
        :return: error
        """
        code: str = 'image_format_error'
        error_message: str = 'There has been an error while formatting the image.'
        return cls(
            code=code,
            error_message=error_message,
            url=url
        )
