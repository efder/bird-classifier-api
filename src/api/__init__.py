# pylint: disable=missing-module-docstring
from typing import List, Any

from flask_restx import Namespace

from src.common.data_objects.birds.bird_name_with_score import BirdNameWithScore
from src.common.errors.base.base_error import BaseError
from src.common.errors.base.base_url_error import BaseUrlError
from src.common.errors.birds.image_download_error import ImageDownloadError
from src.common.errors.birds.image_format_error import ImageFormatError
from src.common.errors.birds.model_inference_error import ModelInferenceError


def create_schemas(api: Namespace) -> List[Any]:
    """
    This method assigns data objects that are used in our namespace with their schemas
    to make their schemas accessible through Swagger UI.
    :param api: Flask namespace
    :return: Schemas
    """
    return [
        # Value objects
        api.schema_model('BaseError', BaseError.schema()),
        api.schema_model('BaseUrlError', BaseUrlError.schema()),
        api.schema_model('BirdNameWithScore', BirdNameWithScore.schema()),
        api.schema_model('ImageDownloadError', ImageDownloadError.schema()),
        api.schema_model('ImageFormatError', ImageFormatError.schema()),
        api.schema_model('ModelInferenceError', ModelInferenceError.schema()),
    ]
