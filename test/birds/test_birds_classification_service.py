from typing import Dict, Union

import numpy as np
from tensorflow_hub import KerasLayer
import tensorflow_hub as hub

from src.common.data_objects.birds.bird_name_with_score import BirdNameWithScore
from src.common.data_objects.birds.birds_classification_response_dto import BirdsClassificationResponseDto
from src.common.data_objects.birds.image_with_url import ImageArrayWithUrl
from src.common.errors.birds.image_download_error import ImageDownloadError
from src.common.errors.birds.image_format_error import ImageFormatError
from src.config import ConfigManager
from src.service.birds.birds_classification_service import BirdClassifier

# Initialize the config
ConfigManager.init_config()
config = ConfigManager.get_config()


def mock_load_and_cleanup_labels() -> Dict[int, Dict[str, str]]:
    with open(config.MOCK_BIRD_LABELS_FILE_PATH) as bird_labels_raw:
        bird_labels_lines = [line.replace('\n', '') for line in bird_labels_raw.readlines()]
        bird_labels_lines.pop(0)  # remove header (id, name)
        birds = {}
        for bird_line in bird_labels_lines:
            bird_id = int(bird_line.split(',')[0])
            bird_name = bird_line.split(',')[1]
            birds[bird_id] = {'name': bird_name}

        return birds


def mock_load_model() -> KerasLayer:
    return hub.KerasLayer(config.MOCK_BIRD_CLASSIFIER_MODEL_PATH)


def mock_download_image(url: str) -> Union[ImageArrayWithUrl, ImageDownloadError]:
    # Check that if the given urls in predefined urls
    predefined_urls = [
        'bird1.jpeg',
        'bird2.jpeg',
        'bird3.jpeg',
        'not_image.txt'
    ]

    # If the given url is not in predefined urls then return an error
    if url not in predefined_urls:
        return ImageDownloadError.from_url(url)

    # Otherwise load the image from our local directory
    with open(f'{config.MOCK_BIRD_IMAGES}/{url}', 'rb') as image:
        image_array = np.asarray(bytearray(image.read()), dtype=np.uint8)
        return ImageArrayWithUrl(
            url=url,
            image_array=image_array
        )


def test_bird_classifier(monkeypatch):
    monkeypatch.setattr(BirdClassifier, '_load_and_cleanup_labels', mock_load_and_cleanup_labels)
    monkeypatch.setattr(BirdClassifier, '_load_model', mock_load_model)
    monkeypatch.setattr(BirdClassifier, 'download_image', mock_download_image)

    BirdClassifier.initialize_bird_classifier()

    # Expected result
    data = {
        'bird1.jpeg': [BirdNameWithScore(bird_name='Phalacrocorax varius varius', score=0.8430764079093933),
                       BirdNameWithScore(bird_name='Phalacrocorax varius', score=0.11654692888259888),
                       BirdNameWithScore(bird_name='Microcarbo melanoleucos', score=0.024331538006663322)],
        'bird2.jpeg': [BirdNameWithScore(bird_name='Galerida cristata', score=0.8428874611854553),
                       BirdNameWithScore(bird_name='Alauda arvensis', score=0.08378683775663376),
                       BirdNameWithScore(bird_name='Eremophila alpestris', score=0.018995530903339386)],
        'bird3.jpeg': [BirdNameWithScore(bird_name='Eumomota superciliosa', score=0.41272449493408203),
                       BirdNameWithScore(bird_name='Momotus coeruliceps', score=0.052539676427841187),
                       BirdNameWithScore(bird_name='Momotus lessonii', score=0.048381607979536057)]
    }

    errors = {}

    expected = BirdsClassificationResponseDto(
        data=data,
        errors=errors
    )

    # Result
    result = BirdClassifier.classify(['bird1.jpeg', 'bird2.jpeg', 'bird3.jpeg'], 3)

    assert expected == result


def test_bird_classifier_image_format_error(monkeypatch):
    monkeypatch.setattr(BirdClassifier, '_load_and_cleanup_labels', mock_load_and_cleanup_labels)
    monkeypatch.setattr(BirdClassifier, '_load_model', mock_load_model)
    monkeypatch.setattr(BirdClassifier, 'download_image', mock_download_image)

    BirdClassifier.initialize_bird_classifier()

    # Expected result
    data = {}

    errors = {'not_image.txt': ImageFormatError.from_url('not_image.txt')}

    expected = BirdsClassificationResponseDto(data=data, errors=errors)

    # Result
    result = BirdClassifier.classify(['not_image.txt'], 3)

    assert expected == result
