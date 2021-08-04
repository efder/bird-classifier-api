"""
Config file which keeps Config and ConfigManager classes that are responsible for managing the variables
that are used along the project.
"""
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """
    Config class
    """
    # Source
    MODEL_URL: str = Field('https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1')
    LABELS_URL: str = Field('https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
    DOWNLOAD_IMAGE_TIMEOUT: int = Field(30)

    # Test
    MOCK_BIRD_LABELS_FILE_PATH: str = Field('test/birds/mock/birds_labelmap.csv')
    MOCK_BIRD_CLASSIFIER_MODEL_PATH: str = Field('test/birds/mock/bird_classifier_model')
    MOCK_BIRD_IMAGES: str = Field('test/birds/mock/images')


class ConfigManager:
    """
    Class responsible for managing life cycle of Config class
    """
    _config: Config

    @classmethod
    def init_config(cls) -> Config:
        """Config initializer"""
        cls._config = Config()
        return cls._config

    @classmethod
    def get_config(cls) -> Config:
        """Config getter"""
        if not cls._config:
            return cls.init_config()
        return cls._config
