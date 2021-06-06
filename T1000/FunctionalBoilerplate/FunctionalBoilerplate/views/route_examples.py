from . import (db_session, posts_tags_table, Comment, Post, Tag, User)

from collections import OrderedDict
from flask import (
    Blueprint,
    render_template,
    request)
from flask.views import (MethodView, View)


class RouteExamplesBlueprintConstants:
    url_prefix = "/routeExamples"

class PostRouteExamplesBlueprintConstants:
    url_prefix = "/postRouteExamples"


route_examples_bp = Blueprint(
    'route_examples_bp',
    __name__,
    url_prefix=RouteExamplesBlueprintConstants.url_prefix)

post_route_examples_bp = Blueprint(
    'route_examples_bp',
    __name__,
    url_prefix=PostRouteExamplesBlueprintConstants.url_prefix)

