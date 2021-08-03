# pylint: disable=missing-module-docstring, import-error
import logging
from typing import Dict, List, Union

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
import urllib.request
import numpy as np
import time
import requests
import concurrent.futures

from tensorflow_hub import KerasLayer

from src.common.data_objects.birds.birds_classification_response_dto import BirdsClassificationResponseDto
from src.common.data_objects.birds.bird_name_with_score import BirdNameWithScore
from src.common.data_objects.birds.image_with_url import ImageArrayWithUrl
from src.common.data_objects.birds.model_raw_output_with_url import ModelRawOutputWithUrl
from src.common.errors.birds.image_download_error import ImageDownloadError
from src.common.errors.birds.image_format_error import ImageFormatError
from src.common.errors.birds.model_inference_error import ModelInferenceError
from src.config import ConfigManager, Config


class BirdClassifier:
    """
    Class which is responsible for bird classification
    """
    _bird_labels: Dict[int, Dict[str, str]]
    _bird_model: KerasLayer
    _logger: logging.Logger
    _config: Config

    @classmethod
    def initialize_bird_classifier(cls) -> None:
        """
        Class initializer method. It is called just once
        :return: None
        """
        cls._config = ConfigManager.get_config()
        cls._logger = logging.getLogger(__name__)
        cls._logger.setLevel(logging.DEBUG)
        start_time = time.time()
        cls._bird_labels = cls._load_and_cleanup_labels()
        bird_labels_end_time = time.time()
        cls._bird_model = cls._load_model()
        bird_model_end_time = time.time()
        cls._logger.debug(f"Labels loaded and cleaned in {bird_labels_end_time - start_time} seconds.")
        cls._logger.debug(f"Bird model loaded in {bird_model_end_time - bird_labels_end_time} seconds.")

    @classmethod
    def _load_model(cls) -> KerasLayer:
        return hub.KerasLayer(cls._config.MODEL_URL)

    @classmethod
    def _load_and_cleanup_labels(cls) -> Dict[int, Dict[str, str]]:
        bird_labels_raw = urllib.request.urlopen(cls._config.LABELS_URL)
        bird_labels_lines = [line.decode('utf-8').replace('\n', '') for line in bird_labels_raw.readlines()]
        bird_labels_lines.pop(0)  # remove header (id, name)
        birds = {}
        for bird_line in bird_labels_lines:
            bird_id = int(bird_line.split(',')[0])
            bird_name = bird_line.split(',')[1]
            birds[bird_id] = {'name': bird_name}

        return birds

    @classmethod
    def get_bird_labels(cls) -> Dict[int, Dict[str, str]]:
        if not cls._bird_labels:
            cls._bird_labels = cls._load_and_cleanup_labels()
        return cls._bird_labels

    @classmethod
    def get_bird_model(cls) -> KerasLayer:
        if not cls._bird_model:
            cls._bird_model = cls._load_model()
        return cls._bird_model

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if not cls._logger:
            cls._logger = logging.getLogger(__name__)
            cls._logger.setLevel(logging.DEBUG)
        return cls._logger

    @classmethod
    def download_image(cls, url: str) -> Union[ImageArrayWithUrl, ImageDownloadError]:
        # Loading images
        try:
            image_get_response = requests.get(url, timeout=cls._config.DOWNLOAD_IMAGE_TIMEOUT)
            if image_get_response.status_code != 200:
                raise Exception('Image can not been fetched!')
            image_array = np.asarray(bytearray(image_get_response.content), dtype=np.uint8)
            return ImageArrayWithUrl(
                url=url,
                image_array=image_array
            )
        except Exception as e:
            cls.get_logger().error(e)
            return ImageDownloadError.from_url(url)

    @classmethod
    def get_top_n_result(cls, top_n: int, model_raw_output_with_url: ModelRawOutputWithUrl,
                         bird_labels: Dict[int, Dict[str, str]]) -> List[BirdNameWithScore]:
        model_raw_output = model_raw_output_with_url.model_raw_output

        # Prevent index out of range
        top_n = min(len(bird_labels), top_n)

        # Get the indices of top k birds without order
        ind = np.argpartition(model_raw_output[0], -top_n)[-top_n:]
        # Sort these indices by their scores and reverse it to make it descending
        sorted_ind = ind[np.argsort(model_raw_output[0][ind])][::-1]

        res = [
            BirdNameWithScore(bird_name=bird_labels[idx]['name'], score=model_raw_output[0][idx])
            for idx in sorted_ind
        ]

        return res

    @classmethod
    def get_model_raw_output(cls, image_array_with_url: ImageArrayWithUrl) -> Union[ModelRawOutputWithUrl,
                                                                                    ImageFormatError,
                                                                                    ModelInferenceError]:
        bird_model = cls.get_bird_model()

        # Changing images
        try:
            image = cv2.imdecode(image_array_with_url.image_array, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255
        except Exception as e:
            cls.get_logger().error(e)
            return ImageFormatError.from_url(image_array_with_url.url)

        # Generate tensor
        try:
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            image_tensor = tf.expand_dims(image_tensor, 0)
            model_raw_output = bird_model.call(image_tensor).numpy()
        except Exception as e:
            cls.get_logger().error(e)
            return ModelInferenceError.from_url(image_array_with_url.url)

        return ModelRawOutputWithUrl(
            url=image_array_with_url.url,
            model_raw_output=model_raw_output
        )

    @classmethod
    def _get_response_log_message(cls,
                                  no_of_image_urls: int,
                                  top_n: int,
                                  no_of_data: int,
                                  no_of_errors: int,
                                  download_image_duration: float,
                                  model_inference_duration: float) -> str:
        return (
            f'\n****************'
            f'\nBirds have been classified.\n'
            f'Request -> no_of_image_urls: {no_of_image_urls}, top_n: {top_n}\n'
            f'Response -> no_of_data: {no_of_data}, no_of_errors: {no_of_errors}\n'
            f'Response time -> download_image: {download_image_duration}, '
            f'model_inference: {model_inference_duration}\n'
            f'****************'
        )

    @classmethod
    def classify(cls, image_urls: List[str], top_n: int) -> BirdsClassificationResponseDto:
        start_time = time.time()

        bird_labels = cls.get_bird_labels()
        errors: Dict[str, Union[ImageDownloadError, ImageFormatError, ModelInferenceError]] = {}
        data: Dict[str,  List[BirdNameWithScore]] = {}

        # Download the images in parallel to prevent I/O block
        with concurrent.futures.ThreadPoolExecutor() as t_executor:
            image_and_error_list = t_executor.map(cls.download_image, image_urls)

        download_image_end_time = time.time()
        image_list = []

        # Filter out the errors
        for item in image_and_error_list:
            # Check that if this an instance of an error
            if isinstance(item, ImageDownloadError):
                errors[item.url] = item
            else:
                image_list.append(item)

        for index, image_array_with_url in enumerate(image_list):
            model_raw_output_or_error = cls.get_model_raw_output(image_array_with_url)
            # Check that if there is an error
            if (isinstance(model_raw_output_or_error, ImageFormatError) or
                    isinstance(model_raw_output_or_error, ModelInferenceError)):
                errors[image_array_with_url.url] = model_raw_output_or_error
            else:
                result = cls.get_top_n_result(top_n, model_raw_output_or_error, bird_labels)
                data[image_array_with_url.url] = result

        model_inference_end_time = time.time()
        download_image_duration = download_image_end_time - start_time
        model_inference_duration = model_inference_end_time - download_image_end_time

        cls.get_logger().debug(
            cls._get_response_log_message(
                no_of_image_urls=len(image_urls),
                top_n=top_n,
                no_of_data=len(data),
                no_of_errors=len(errors),
                download_image_duration=download_image_duration,
                model_inference_duration=model_inference_duration
            )
        )

        return BirdsClassificationResponseDto(
            data=data,
            errors=errors
        )
