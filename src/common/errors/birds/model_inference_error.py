from __future__ import annotations

from src.common.errors.base.base_url_error import BaseUrlError


class ModelInferenceError(BaseUrlError):
    @classmethod
    def from_url(cls, url: str) -> ModelInferenceError:
        code: str = 'model_inference_error'
        error_message: str = 'There has been an error while classifying the image.'
        return cls(
            code=code,
            error_message=error_message,
            url=url
        )