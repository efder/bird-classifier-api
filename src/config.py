from pydantic import BaseSettings, Extra, Field


class Config(BaseSettings):
    # Source
    MODEL_URL: str = Field('https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1')
    LABELS_URL: str = Field('https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
    DOWNLOAD_IMAGE_TIMEOUT: int = Field(10)

    # Test
    MOCK_BIRD_LABELS_FILE_PATH: str = Field('mock/birds_labelmap.csv')
    MOCK_BIRD_CLASSIFIER_MODEL_PATH: str = Field('mock/bird_classifier_model')
    MOCK_BIRD_IMAGES: str = Field('mock/images')

    class Config:
        extra = Extra.allow


class ConfigManager:
    _config: Config

    @classmethod
    def init_config(cls) -> Config:
        cls._config = Config()
        return cls._config

    @classmethod
    def get_config(cls) -> Config:
        if not cls._config:
            return cls.init_config()
        return cls._config
