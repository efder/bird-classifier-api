# pylint: disable=missing-module-docstring, missing-function-docstring
from typing import Optional

from flask import Flask
from flask_restx import Api

from src.api.controllers.birds import birds_api
from src.api.controllers.birds.birds_classification_controller import api as bird_classification_api


class APIManager:
    """
    API manager which is responsible for managing different namespaces.
    """
    def __init__(self, app: Flask) -> None:
        self.birds_api: Optional[Api] = None
        self.init_apis(app)

    def init_apis(self, app: Flask) -> None:
        self.birds_api = Api(
            birds_api,
            title='Bird API',
            version='0.0.1',
            description='API for birds classification'
        )

        # Register namespaces
        self.birds_api.add_namespace(bird_classification_api)

        # Register blueprints
        app.register_blueprint(birds_api)
