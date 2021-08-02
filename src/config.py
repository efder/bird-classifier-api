from pydantic import BaseSettings, Extra, Field


class Config(BaseSettings):
    MODEL_URL: str = Field('https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1')
    LABELS_URL: str = Field('https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
    DOWNLOAD_IMAGE_TIMEOUT: int = Field(10)

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
