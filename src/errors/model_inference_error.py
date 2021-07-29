from src.errors.base_url_error import BaseUrlError


class ModelInferenceError(BaseUrlError):
    def __init__(self, url: str) -> None:
        super().__init__(
            code='model_inference_error',
            error_message='There has been an error while model was applying inference on your image.',
            url=url
        )
