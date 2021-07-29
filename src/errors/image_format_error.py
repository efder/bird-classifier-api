from src.errors.base_url_error import BaseUrlError


class ImageFormatError(BaseUrlError):
    def __init__(self, url: str) -> None:
        super().__init__(
            code='image_format_error',
            error_message='There has been an error while formatting your image.',
            url=url
        )
