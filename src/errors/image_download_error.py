from src.errors.base_url_error import BaseUrlError


class ImageDownloadError(BaseUrlError):
    def __init__(self, url: str) -> None:
        super().__init__(
            code='image_download_error',
            error_message='There has been an error while downloading an image.',
            url=url
        )
