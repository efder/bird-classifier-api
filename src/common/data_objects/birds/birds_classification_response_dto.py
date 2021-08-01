from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from src.common.data_objects.birds.bird_name_with_score import BirdNameWithScore
from src.common.errors.birds.image_download_error import ImageDownloadError
from src.common.errors.birds.image_format_error import ImageFormatError
from src.common.errors.birds.model_inference_error import ModelInferenceError


class BirdsClassificationResponseDto(BaseModel):
    data: Optional[Dict[str, List[BirdNameWithScore]]]
    errors: Optional[Dict[str, Union[ImageDownloadError, ImageFormatError, ModelInferenceError]]]
