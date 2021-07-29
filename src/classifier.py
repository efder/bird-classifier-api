import os
from typing import Dict, List, Tuple, Union

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
import urllib.request
import numpy as np
import time
import concurrent.futures

# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all
from numpy import ndarray
from tensorflow_hub import KerasLayer

from src.data_objects.bird_classification_response_dto import BirdClassificationResponseDto
from src.data_objects.bird_name_with_score import BirdNameWithScore
from src.data_objects.image_with_url import ImageArrayWithUrl
from src.data_objects.model_raw_output_with_url import ModelRawOutputWithUrl
from src.errors.image_download_error import ImageDownloadError
from src.errors.image_format_error import ImageFormatError
from src.errors.model_inference_error import ModelInferenceError

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow logging

model_url = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'
labels_url = 'https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv'

image_urls = [
    #'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    #'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    #'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    #'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    'https://cdn.britannica.co/77/189277-004-0A3BC3D4.jpg',
]


class BirdClassifier:
    _bird_labels = None
    _bird_model = None

    @classmethod
    def initialize_bird_classifier(cls) -> None:
        cls._bird_labels = cls._load_and_cleanup_labels()
        cls._bird_model = cls._load_model()

    @classmethod
    def _load_model(cls) -> KerasLayer:
        return hub.KerasLayer(model_url)

    @classmethod
    def _load_and_cleanup_labels(cls) -> Dict[int, Dict[str, str]]:
        bird_labels_raw = urllib.request.urlopen(labels_url)
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
    def download_image(cls, url: str) -> Union[ImageArrayWithUrl, ImageDownloadError]:
        # Loading images
        try:
            image_get_response = urllib.request.urlopen(url)
            if image_get_response.status != 200:
                raise Exception('Image can not been fetched!')
            image_array = np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)
            return ImageArrayWithUrl(
                url=url,
                image_array=image_array
            )
        except Exception as e:
            # TODO: log the exception, but print for now
            print(e)
            return ImageDownloadError(url)

    @classmethod
    def get_top_n_result(cls, top_n: int, model_raw_output_with_url: ModelRawOutputWithUrl,
                         bird_labels: Dict[int, Dict[str, str]]) -> List[BirdNameWithScore]:
        model_raw_output = model_raw_output_with_url.model_raw_output

        # TODO: optimize this part
        # Prevent index out of range
        top_n = min(len(bird_labels), top_n)

        ind = np.argpartition(model_raw_output[0], -top_n)
        sorted_ind = ind[np.argsort(model_raw_output[0][ind])][::-1][:top_n]
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
            # TODO: log the exception, but print for now
            print(e)
            return ImageFormatError(image_array_with_url.url)

        # Generate tensor
        try:
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            image_tensor = tf.expand_dims(image_tensor, 0)
            model_raw_output = bird_model.call(image_tensor).numpy()
        except Exception as e:
            # TODO: log the exception, but print for now
            print(e)
            return ModelInferenceError(image_array_with_url.url)

        return ModelRawOutputWithUrl(
            url=image_array_with_url.url,
            model_raw_output=model_raw_output
        )

    @classmethod
    def print_result(cls, index: int, result: List[BirdNameWithScore]) -> None:
        print('Run: %s' % int(index + 1))
        bird_name, bird_score = result[0].bird_name, result[0].score
        print('Top match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = result[1].bird_name, result[1].score
        print('Second match: "%s" with score: %s' % (bird_name, bird_score))
        bird_name, bird_score = result[2].bird_name, result[2].score
        print('Third match: "%s" with score: %s' % (bird_name, bird_score))
        print('\n')

    @classmethod
    def get_result(cls) -> BirdClassificationResponseDto:
        bird_labels = cls.get_bird_labels()
        errors = {}
        data = {}

        with concurrent.futures.ThreadPoolExecutor() as t_executor:
            image_and_error_list = t_executor.map(cls.download_image, image_urls)

        image_list = []

        # Filter out the errors
        for item in image_and_error_list:
            # Check that if this an instance of an error
            if isinstance(item, ImageDownloadError):
                errors[item.url] = item
            else:
                image_list.append(item)

        print('Time spent: %s - after download' % (time.time() - start_time))

        for index, image_array_with_url in enumerate(image_list):
            model_raw_output_or_error = cls.get_model_raw_output(image_array_with_url)
            # Check that if there is an error
            if (isinstance(model_raw_output_or_error, ImageFormatError) or
                    isinstance(model_raw_output_or_error, ModelInferenceError)):
                errors[image_array_with_url.url] = model_raw_output_or_error
            else:
                result = cls.get_top_n_result(3, model_raw_output_or_error, bird_labels)
                data[image_array_with_url.url] = result

        print('Time spent: %s - after model' % (time.time() - start_time))

        return BirdClassificationResponseDto(
            data=data,
            errors=errors
        )


if __name__ == "__main__":
    BirdClassifier.initialize_bird_classifier()
    start_time = time.time()
    print(BirdClassifier.get_result().dict())
    print('Time spent: %s' % (time.time() - start_time))
