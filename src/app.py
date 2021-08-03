"""
Flask application initialization module
"""
from flask import Flask

from src.config import ConfigManager
from src.api.helpers.api_manager import APIManager
from src.service.birds.birds_classification_service import BirdClassifier
from src.api.helpers.error_handler import ErrorHandler


def create_app() -> Flask:
    """
     Application factory function that is called by flask only one time. That is
     very useful for initializing configurations and services before application starts.

     :return: None
    """
    app = Flask(__name__)

    # Initialize config variable
    ConfigManager.init_config()

    # Initialize apis
    APIManager(app)

    # Initialize services
    BirdClassifier.initialize_bird_classifier()

    # Initialize error handler
    ErrorHandler.initialize(app)

    return app
