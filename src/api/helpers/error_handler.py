# pylint: disable=missing-module-docstring,missing-function-docstring, invalid-name
import logging
from http import HTTPStatus
from typing import Any

from flask import Response, Flask
from pydantic import ValidationError
from werkzeug.exceptions import NotFound

logger = logging.getLogger(__name__)


def handle_message(e: Any) -> str:
    if not e.args:
        return ''
    if isinstance(e.args[0], list):
        return ', '.join(e.args[0])
    return ', '.join(e.args)


def handle_validation_error_message(error: ValidationError) -> str:
    message = 'One or more validation errors occurred:\n'
    errors = '\n'.join([f'{e.get("loc", "")}: {e.get("msg", "")}' for e in error.errors()])

    return message + errors


def handle_validation_error(validation_error: ValidationError) -> Response:
    # Validation errors are caused by invalid input, BadRequest should be returned
    message = handle_validation_error_message(validation_error)
    logger.error('Validation Error caught. Message: %s',
                 message,
                 exc_info=True)
    return Response(
        response=message,
        status=HTTPStatus.BAD_REQUEST
    )


def handle_not_found_error(not_found_error: NotFound) -> Response:
    # Validation errors are caused by invalid input, BadRequest should be returned
    message = handle_message(not_found_error)
    return Response(
        response=message,
        status=HTTPStatus.NOT_FOUND
    )


def handle_exception(exception: Exception) -> Response:
    # Unhandled exceptions should be returned with InternalServerError
    message = handle_message(exception)
    logger.critical('Unhandled Error caught. Message: %s',
                    message,
                    exc_info=True)
    return Response(
        response='Something unexpected happened. Try later.',
        status=HTTPStatus.INTERNAL_SERVER_ERROR
    )


class ErrorHandler:
    """
    Error handler class which is responsible for assigning error_handlers to expected errors at start
    """
    @classmethod
    def initialize(cls, app: Flask) -> None:
        app.register_error_handler(ValidationError, handle_validation_error)  # type: ignore
        app.register_error_handler(NotFound, handle_not_found_error)  # type: ignore
        app.register_error_handler(Exception, handle_exception)
