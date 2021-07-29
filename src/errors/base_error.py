from pydantic import BaseModel


class BaseError(BaseModel):
    code: str
    error_message: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
