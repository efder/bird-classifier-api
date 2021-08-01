import logging

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    # Initialize apis
    from src.api.helpers.api_manager import APIManager
    APIManager(app)

    # Initialize services
    from src.service.birds.birds_classification_service import BirdClassifier
    BirdClassifier.initialize_bird_classifier()

    # Initialize error handler
    from src.api.helpers.error_handler import ErrorHandler
    ErrorHandler.initialize(app)

    return app
