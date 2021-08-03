"""
Namespace to keep all bird classification related endpoints.
"""
# pylint: disable=no-self-use
from http import HTTPStatus

from flask import Response, request
from flask_restx import Namespace, Resource

from src.api import create_schemas
from src.common.data_objects.birds.birds_classification_request_dto import BirdsClassificationRequestDto
from src.common.data_objects.birds.birds_classification_response_dto import BirdsClassificationResponseDto
from src.service.birds.birds_classification_service import BirdClassifier

api = Namespace('classification', description='API for birds classification')

# request dtos
birds_classification_request = api.schema_model('birds_classification_request_dto',
                                                BirdsClassificationRequestDto.schema())

# response dtos
birds_classification_response = api.schema_model('birds_classification_response_dto',
                                                 BirdsClassificationResponseDto.schema())

used_schemas = create_schemas(api)


@api.route('')
class GetBirdsClassification(Resource):
    """
    Flask REST resource for serving birds classification results.
    """
    @api.doc(description="Returns top n classifications for the given bird image urls.")
    @api.expect(*used_schemas, birds_classification_request)
    @api.response(200, 'Birds classification result', birds_classification_response)
    def post(self) -> Response:
        """
        POST endpoint for GetBirdsClassification resource
        :return: Http response
        """
        data = request.get_json()
        request_dto = BirdsClassificationRequestDto.parse_obj(data)

        response_dto = BirdClassifier.classify(request_dto.urls, request_dto.top_n)

        return Response(response=response_dto.json(), status=HTTPStatus.OK)
