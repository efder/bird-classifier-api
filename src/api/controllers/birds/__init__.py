from flask import Blueprint

birds_api = Blueprint('birds_api', __name__, url_prefix='/birds/api/v1')
