import datetime

from flask import abort, current_app, jsonify, request
from flask_restful import Resource, fields, marshal_with
#from flask_jwt_extended import jwt_required, get_jwt_identity

from FunctionalBoilerplate.models import (Post, Tag)
from FunctionalBoilerplate.setup_database_session import db_session