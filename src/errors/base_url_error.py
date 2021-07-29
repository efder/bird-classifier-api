from src.errors.base_error import BaseError


class BaseUrlError(BaseError):
    url: str

    def __init__(self, *args, **kwargs):
        super(BaseUrlError, self).__init__(*args, **kwargs)
        self.url = kwargs['url']
