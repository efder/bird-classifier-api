# pylint: disable=missing-module-docstring
from __future__ import annotations

from src.common.errors.base.base_url_error import BaseUrlError


class ModelInferenceError(BaseUrlError):
    """
    Basic data object to represent any error happens while model applying inference on an image.
    """
    @classmethod
    def from_url(cls, url: str) -> ModelInferenceError:
        """
        Creates ModelInferenceError using url
        :param url: requested_url
        :return: error
        """
        code: str = 'model_inference_error'
        error_message: str = 'There has been an error while classifying the image.'
        return cls(
            code=code,
            error_message=error_message,
            url=url
        )
